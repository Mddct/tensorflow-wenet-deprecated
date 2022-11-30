import tensorflow as tf
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTCDense
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.utils.cmvn import load_cmvn


def init_model(configs, if_model_summary=True):
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(tf.convert_to_tensor(mean, dtype=tf.float32),
                                 tf.convert_to_tensor(istd, dtype=tf.float32))
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')

    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim,
                                   global_cmvn=global_cmvn,
                                   **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])

    ctc_weight = configs['model_conf']['ctc_weight']
    decoder = None
    reverse_decoder = None
    if ctc_weight != 1.0:
        decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                     **configs['decoder_conf'])
        if decoder_type == 'Bitransformer':
            assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
            assert configs['decoder_conf']['r_num_blocks'] > 0
            reverse_decoder = TransformerDecoder(vocab_size,
                                                 encoder.output_size(),
                                                 **configs['decoder_conf'])
    ctc_dense = None
    if ctc_weight != 0.0:
        ctc_dense = CTCDense(vocab_size)

    # Init joint CTC/Attention or Transducer model
    if 'predictor' in configs:
        raise NotImplementedError("transducer not suport now")
    else:
        model = ASRModel(vocab_size=vocab_size,
                         encoder=encoder,
                         decoder=decoder,
                         ctcdense=ctc_dense,
                         reverse_decoder=reverse_decoder,
                         **configs['model_conf'])
    # force build model here
    inputs = [
        tf.keras.layers.Input([None, 80], dtype="float32", name="feats"),
        tf.keras.layers.Input((), dtype=tf.int32, name="feats_lens"),
        tf.keras.layers.Input((None, ), dtype=tf.int32, name="labels"),
        tf.keras.layers.Input((), dtype=tf.int32, name="labels_length"),
    ]
    model(inputs)
    if if_model_summary:
        model.summary()

    return model
