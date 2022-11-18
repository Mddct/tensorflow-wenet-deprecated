from numpy import dtype
import tensorflow as tf
import yaml

from absl import app, flags

from wenet.utils.init_model import init_model

FLAGS = flags.FLAGS

flags.DEFINE_string('config', default=None, required=True, help='config file')
flags.DEFINE_string('output_path',
                    default=None,
                    required=True,
                    help='output savedmodel path')
flags.DEFINE_string('checkpoint',
                    default=None,
                    required=True,
                    help='checkpoint')


def main(argv):
    with open(FLAGS.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    vocab_size = 4233
    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size

    model = init_model(configs)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(FLAGS.checkpoint)

    # TODO: different chunk_size
    module_wenet = Decoder(model, 16, 16)
    chunk_outs = module_wenet.forward_encoder_chunk.get_concrete_function(
        tf.TensorSpec([1, None, None], dtype=tf.float32),
        tf.TensorSpec([None], dtype=tf.int32),
        tf.TensorSpec([None], dtype=tf.int32),
        tf.TensorSpec([None], dtype=tf.int32),
        tf.TensorSpec([None, None, None, None], dtype=tf.float32),
        tf.TensorSpec([None, None, None, None], dtype=tf.float32),
    )
    metadata = module_wenet.metadata.get_concrete_function(
        tf.TensorSpec(None, tf.bool))
    tf.saved_model.save(module_wenet,
                        FLAGS.output_path,
                        signatures={
                            'forward_encoder_chunk': chunk_outs,
                            "metadata": metadata
                        })


if __name__ == '__main__':
    app.run(main)
