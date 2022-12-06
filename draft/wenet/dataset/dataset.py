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
    data_list_file,
    global_batch_size=1,
    prefetch=tf.data.AUTOTUNE,
    data_type="shard",
    strategy=None,
    training: bool = True,
    cache: bool = False,
    tf_data_service: Optional[str] = None,
):

    symbol_table, vocab_size = look_up_table(symbol_table_path)

    def dataset_fn(input_context=None):
        dataset = tf.data.TextLineDataset(data_list_file)
        if data_type == 'shard':
            # eacho shard: dataset element ["shard1.txt", 'shard2.txt'....]
            # asynchronously parallel read multiple shard
            dataset = dataset.interleave(
                lambda elem: tf.data.TextLineDataset(elem),
                cycle_length=prefetch,
                num_parallel_calls=prefetch,
            )
            dataset = dataset.apply(tf.data.experimental.ignore_errors())

        if strategy is not None and input_context is not None:
            dataset = dataset.shard(input_context.num_input_pipelines,
                                    input_context.input_pipeline_id)
        shuffle = conf.get('shuffle', True)
        if shuffle:
            shuffle_conf = conf.get('shuffle_conf', {})
            dataset = dataset.shuffle(buffer_size=shuffle_conf['shuffle_size'],
                                      reshuffle_each_iteration=True)
        if training and cache is not None:
            dataset = dataset.repeat()
            shard = 1000
            prefetch_wav = 100
            dataset = dataset.window(shard)

            dataset = dataset.interleave(
                lambda window: window.map(lambda line: processor.parse_line(
                    line, symbol_table),
                                          num_parallel_calls=tf.data.AUTOTUNE).
                apply(tf.data.experimental.ignore_errors()),
                num_parallel_calls=tf.data.AUTOTUNE,
                block_length=prefetch_wav,
            )

        if cache:
            # for small dataset we can cache all raw wav in memory
            dataset = dataset.cache()

        # dataset = dataset.map(
        #     lambda line: processor.parse_line(line, symbol_table),
        #     num_parallel_calls=tf.data.AUTOTUNE)
        # file may not found in  parse_line, ignore error
        # dataset = dataset.apply(tf.data.experimental.ignore_errors())

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
        bucket_boundaries = [500, 10 * 100, 15 * 100,
                             20 * 100]  # [0-5s) [5s, 10s) [10s,15) ...
        bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)
        dataset = dataset.bucket_by_sequence_length(
            lambda inputs: tf.cast(inputs[1], tf.int32),
            bucket_boundaries,
            bucket_batch_sizes,
            pad_to_bucket_boundary=False,
            drop_remainder=True,
            padded_shapes=([None, None], [], [None], []),
            # TODO: ignore id not 0 for text padding id
            padding_values=(0.0, None, tf.cast(0, dtype=tf.int32), None),
        )
        # group by window
        if input_context is not None:
            window_size = input_context.num_replicas_in_sync
            dataset = dataset.group_by_window(
                key_func=lambda inputs: tf.cast(  # pylint: disable=g-long-lambda
                    tf.shape(inputs[0])[1], tf.int64),
                reduce_func=lambda inputs:
                (tf.data.Dataset.from_tensors(inputs[0]),
                 tf.data.Dataset.from_tensors(inputs[1]),
                 tf.data.Dataset.from_tensors(inputs[2]),
                 tf.data.Dataset.from_tensors(inputs[3])),
                window_size=window_size)
            dataset = dataset.flat_map(lambda x: x)

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
