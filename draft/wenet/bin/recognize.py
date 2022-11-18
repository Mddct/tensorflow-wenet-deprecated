import tensorflow as tf
import yaml

from absl import app, flags

from wenet.utils.init_model import init_model
from wenet.transformer.search.ctc_search import CTCSearch
from wenet.transformer.inference import Decoder
from wenet.dataset.dataset import Dataset
import copy

FLAGS = flags.FLAGS

flags.DEFINE_string('cmvn', default=None, help='global cmvn file')
flags.DEFINE_string('config', default=None, required=True, help='config file')
flags.DEFINE_string('checkpoint',
                    default=None,
                    required=True,
                    help='checkpoint')
flags.DEFINE_string('wav_path', default=None, required=True, help='wav file')
flags.DEFINE_enum('mode',
                  enum_values=['ctc_greedy_search', 'ctc_prefix_beam_search'],
                  default='ctc_prefix_beam_search',
                  help='decoding mode')
flags.DEFINE_enum('data_type',
                  default='raw',
                  enum_values=['raw', 'shard'],
                  help='data type')
flags.DEFINE_string('data_list',
                    default=None,
                    required=True,
                    help='eval data file')
flags.DEFINE_enum('mode',
                  enum_values=[
                      'attention',
                      'ctc_greedy_search',
                      'ctc_prefix_beam_search',
                      'attention_rescoring',
                      'rnnt_greedy_search',
                      'rnnt_beam_search',
                      'rnnt_beam_attn_rescoring',
                  ],
                  default='attention',
                  help='decoding mode')


def main(argv):
    with open(FLAGS.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']

    dataset_conf = configs['dataset_conf']
    global_batch_size = dataset_conf['batch_conf']['batch_size']
    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['spec_sub'] = False
    test_conf['spec_trim'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    if 'fbank_conf' in test_conf:
        test_conf['fbank_conf']['dither'] = 0.0
        test_conf['batch_conf']['batch_type'] = "static"
        # TODO: fix eval dataset
    dataset, vocab_size = Dataset(
        dataset_conf,
        FLAGS.data_list,
        global_batch_size,
        training=False,
        data_type=FLAGS.data_type,
    )

    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = FLAGS.cmvn
    configs['is_json_cmvn'] = True

    model = init_model(configs)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(FLAGS.checkpoint)

    module_wenet = Decoder(model, -1, -1)
    mode = FLAGS.mode
    if 'ctc' in mode:
        search = CTCSearch(FLAGS.beam)
    else:
        raise NotImplementedError('only ctc search support now')
    for batch in dataset:
        feats, feats_lens = batch
        chunks_out = module_wenet.forward_encoder(
            feats,
            feats_lens,
        )
        if mode == 'ctc_greedy_search':
            ctc_activation, lens = chunks_out['ctc_activation'], chunks_out[
                'encoder_out_lens']
            decodes, neg_sum_logits = search.greedy_search(
                ctc_activation, lens)
        elif mode == 'ctc_beam_search':
            ctc_activation, lens = chunks_out['ctc_activation'], chunks_out[
                'encoder_out_lens']
            decodes, neg_sum_logits = search.greedy_search(
                ctc_activation, lens)
        else:
            pass
        # TODO: convert ids to chars


if __name__ == '__main__':
    app.run(main)
