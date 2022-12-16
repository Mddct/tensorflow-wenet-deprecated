from typing import Optional
import tensorflow as tf
import wenet.dataset.processor as processor
from wenet.tfaudio import SpectrumAugmenter
from wenet.utils.file_utils import read_symbol_table


def look_up_table(symbol_table_path, unk="<unk>"):

    words, ids = read_symbol_table(symbol_table_path)
    init = tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(words, dtype=tf.string),
        values=tf.constant(ids, dtype=tf.int32),
    )
    unk_id = ids[words.index(unk)]
    return tf.lookup.StaticHashTable(
        init,
        default_value=unk_id,
    ), len(words)


def Dataset(
    conf,
    symbol_table_path,
    file_pattern,
    global_batch_size=1,
    strategy=None,
    prefetch=1000,
    training: bool = True,
    cache: bool = False,
    max_io_parallelism: int = 256,
    tf_data_service: Optional[str] = None,
):

    symbol_table, vocab_size = look_up_table(symbol_table_path)

    def _load_data_list(filename):
        return tf.data.TextLineDataset(filename)

    def dataset_fn(input_context=None):
        if training:
            dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
        else:
            dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
            # shard across multi workers
        if strategy is not None and input_context is not None:
            dataset = dataset.shard(input_context.num_input_pipelines,
                                    input_context.input_pipeline_id)

        # Read files and interleave results. When training, the order of the examples
        # will be non-deterministic.
        options = tf.data.Options()
        if training:
            options.deterministic = False
        else:
            options.deterministic = True

        dataset = dataset.interleave(
            _load_data_list,
            cycle_length=max_io_parallelism,
            num_parallel_calls=tf.data.AUTOTUNE).with_options(options)

        shuffle = conf.get('shuffle', True)
        if shuffle:
            shuffle_conf = conf.get('shuffle_conf', {})
            dataset = dataset.shuffle(buffer_size=shuffle_conf['shuffle_size'],
                                      reshuffle_each_iteration=True)
        if training:
            # for training , repeat forwever, until reach max steps
            dataset = dataset.repeat()
            # interleave wav read
            dataset = dataset.interleave(
                lambda elem: tf.data.Dataset.from_tensors(elem).map(
                    lambda line: processor.parse_line(line, symbol_table),
                    num_parallel_calls=1).ignore_errors(),
                cycle_length=prefetch,
                num_parallel_calls=max_io_parallelism,
            )

        if cache:
            # for small dataset we can cache all raw wav in memory
            dataset = dataset.cache()

        # 1 one sample, like: resample speed 、spec sub、 spec trim、filter ...
        # 2 fbank
        # 3 group by window
        # 5 batch sample, like: spec aug
        speed_perturb = conf.get('speed_perturb', False)
        if speed_perturb:
            dataset = dataset.map(
                lambda waveform, sr, labels: (processor.speed_perturb(
                    waveform, sr, tf.constant([0.9, 1.0, 1.1])), sr, labels),
                tf.data.AUTOTUNE)

        filter_conf = conf.get('filter_conf', {})
        dataset = dataset.filter(lambda waveform, sr, labels: (
            processor.filter(waveform, sr, labels, **filter_conf)))

        fbank_conf = conf.get('fbank_conf', {})
        dataset = dataset.map(
            lambda waveform, sr, labels:
            (processor.compute_fbank(waveform, sr, **fbank_conf), labels),
            num_parallel_calls=tf.data.AUTOTUNE)

        spec_trim = conf.get('spec_trim', False)
        if spec_trim:
            spec_trim_conf = conf.get('spec_trim_conf', {})
            dataset = dataset.map(
                lambda feats, labels:
                (processor.spec_trim(feats, **spec_trim_conf), labels),
                num_parallel_calls=tf.data.AUTOTUNE)

        # get feats_length, labels_length
        dataset = dataset.map(
            lambda feats, labels:
            (feats, tf.shape(feats)[0], labels, tf.shape(labels)[0]),
            tf.data.AUTOTUNE)  # feats, feats_length, labels, labels_length
        batch_size = global_batch_size
        if strategy is not None:
            batch_size = input_context.get_per_replica_batch_size(
                global_batch_size)
            # bucket
            # TODO: from config
            # [0-1s)... [5s, 10s) [10s,15) ...
        bucket_boundaries = [5 * 100, 10 * 100, 15 * 100, 20 * 100]
        bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        # TODO: replace with gorup by window, each batch has similar total tokens
        dataset = dataset.bucket_by_sequence_length(
            lambda feats, feats_lens, labels, labels_lens: feats_lens,
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padded_shapes=([None, None], [], [None], []),
            padding_values=(0.0, None, tf.cast(0, dtype=tf.int32), None),
            drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        # dataset = dataset.padded_batch(batch_size=batch_size,
        #                                padded_shapes=([None,
        #                                                None], [], [None], []),
        #                                padding_values=(0.0, None,
        #                                                tf.cast(0,
        #                                                        dtype=tf.int32),
        #                                                None),
        #                                drop_remainder=True)

        # batch spec aug
        spec_aug = conf.get('spec_aug', True)
        if spec_aug:
            spec_aug_conf = conf.get('spec_aug_conf', {})
            augmenter = SpectrumAugmenter()
            augmenter.params['time_mask_count'] = spec_aug_conf['num_t_mask']
            augmenter.params['freq_mask_count'] = spec_aug_conf['num_f_mask']
            augmenter.params['time_mask_max_frames'] = spec_aug_conf['max_t']
            augmenter.params['freq_mask_max_bins'] = spec_aug_conf['max_f']
            dataset = dataset.map(
                lambda feats, feats_length, labels, labels_length:
                (processor.spec_aug(feats, feats_length, augmenter),
                 feats_length, labels, labels_length), tf.data.AUTOTUNE)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        if tf_data_service is not None:
            if not hasattr(tf.data.experimental, 'service'):
                raise ValueError(
                    'The tf_data_service flag requires Tensorflow version '
                    '>= 2.3.0, but the version is {}'.format(tf.__version__))
            dataset = dataset.apply(
                tf.data.experimental.service.distribute(
                    processing_mode='distributed_epoch',
                    service=tf_data_service,
                    job_name='wenet_train'))
            dataset = dataset.prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    if strategy is None:
        return dataset_fn(None), vocab_size

    return strategy.distribute_datasets_from_function(dataset_fn), vocab_size


# config = "../bin/conformer.yaml"
# import yaml

# with open(config, 'r') as fin:
#     configs = yaml.load(fin, Loader=yaml.FullLoader)

# input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']

# configs['input_dim'] = input_dim
# configs['output_dim'] = 5000
# configs['cmvn_file'] = None
# configs['is_json_cmvn'] = True

# dataset = Dataset(configs['dataset_conf'],
#                   "train.txt",
#                   data_type='raw',
#                   global_batch_size=2)

# for feats, feats_length, labels, labels_length in dataset:
#     print(feats.shape, feats_length, labels.shape, labels_length)
