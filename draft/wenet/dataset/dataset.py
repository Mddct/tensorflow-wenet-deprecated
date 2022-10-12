import tensorflow as tf
import wenet.dataset.processor as processor

# communication_options = tf.distribute.experimental.CommunicationOptions(
#     implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)

# strategy = tf.distribute.MultiWorkerMirroredStrategy(
#     communication_options=communication_options)

# strategy = tf.distribute.MirroredStrategy()

# dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)


def read_lists(list_file):
    lists = []
    with open(list_file, 'r', encoding='utf8') as fin:
        for line in fin:
            lists.append(line.strip())
    return lists


def Dataset(conf,
            data_list_file,
            global_batch_size=1,
            prefetch=tf.data.AUTOTUNE,
            data_type="shard",
            strategy=None):

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
        shuffle = conf.get('shuffle', True)
        if shuffle:
            shuffle_conf = conf.get('shuffle_conf', {})
            dataset = dataset.shuffle(buffer_size=shuffle_conf['shuffle_size'],
                                      reshuffle_each_iteration=True)

        if strategy is not None:
            dataset = dataset.shard(input_context.num_input_pipelines,
                                    input_context.input_pipeline_id)

        dataset = dataset.map(processor.parse_line,
                              num_parallel_calls=tf.data.AUTOTUNE)
        # file may not found in  parse_line, ignore error
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        dataset = dataset.map(
            lambda waveform, sr, labels:
            (processor.speed_perturb(waveform, sr), sr, labels),
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
                lambda waveform, labels:
                (processor.spec_trim(waveform, **spec_trim_conf), labels),
                num_parallel_calls=tf.data.AUTOTUNE)

        batch_size = global_batch_size
        if strategy is not None:
            batch_size = input_context.get_per_replica_batch_size(
                global_batch_size)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=([None, None], [None]),
            padding_values=(0.0, tf.cast(0, dtype=tf.int64)),
            drop_remainder=True)
        return dataset

    if strategy is None:
        return dataset_fn()

    return strategy.distribute_datasets_from_function(dataset_fn)


# config = "../bin/conformer.yaml"
# import yaml

# with open(config, 'r') as fin:
#     configs = yaml.load(fin, Loader=yaml.FullLoader)

# input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']

# configs['input_dim'] = input_dim
# configs['output_dim'] = 5000
# configs['cmvn_file'] = None
# configs['is_json_cmvn'] = True

# dataset = Dataset(configs['dataset_conf'], "train.txt", data_type='raw')

# for feats, labels in dataset:
#     print(feats.shape, labels.shape)
