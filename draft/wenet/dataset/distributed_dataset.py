import tensorflow as tf
import tensorflow_io as tfio

communication_options = tf.distribute.experimental.CommunicationOptions(
    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)

# strategy = tf.distribute.MultiWorkerMirroredStrategy(
#     communication_options=communication_options)


strategy = tf.distribute.MirroredStrategy()
# dataset = tf.data.Dataset.from_tensor_slices(wav_list)
def dataset_fn(input_context):
    # a tf.data.Dataset
    dataset = tf.data.TextLineDataset("test.csv")
    dataset = dataset.shard(
        input_context.num_input_pipelines,
        input_context.input_pipeline_id)

    def decode(path):
        wav, _ = tf.audio.decode_wav(path)
        return wav
    dataset = dataset.map(decode)
    # Custom your batching, sharding, prefetching, etc.
    global_batch_size = 10
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    dataset = dataset.batch(batch_size)
    return dataset

dist_dataset = strategy.distribute_datasets_from_function(dataset_fn)
