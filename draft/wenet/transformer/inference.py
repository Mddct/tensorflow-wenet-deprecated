import tensorflow as tf


class Decoder(tf.Module):

    def __init__(self,
                 model: tf.keras.Model,
                 chunk_size=-1,
                 num_decoding_left_chunks=-1):
        super(Decoder, self).__init__()
        self.model = model
        self.chunk_size = chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

    def forward_encoder(
        self,
        xs: tf.Tensor,
        xs_lens: tf.Tensor,
    ):
        """ python inference"""
        encoder_out, encoder_out_mask = self.model.encoder(
            [xs, xs_lens],
            self.chunk_size,
            self.num_decoding_left_chunks,
            training=False)
        encoder_out_lens = tf.reduce_sum(tf.cast(tf.squeeze(encoder_out_mask),
                                                 dtype=tf.int32),
                                         axis=-1)
        ctc_log_probs = self.model.ctc_dense.log_softmax(encoder_out)
        return {
            "encoder_out": encoder_out,
            "encoder_out_lens": encoder_out_lens,
            "ctc_activation": ctc_log_probs,
        }

    def attention_rescoring(
        self,
        hyps: tf.Tensor,
        hyps_lens: tf.Tensor,
        encoder_out: tf.Tensor,
        reverse_weight: float = 0,
    ):
        """python inference"""
        pass

    def _get_symbols_to_logits_fn(self, max_decode_length):
        """Returns a decoding function that calculates logits of the next tokens.
           python inference
        """
        decoder = self.model.decoder
        return decoder._get_symbos_to_logits_fn(max_decode_length,
                                                training=False)

    @tf.function(input_signature=[
        tf.TensorSpec([1, None, None], dtype=tf.float32),
        tf.TensorSpec([None], dtype=tf.int32),
        tf.TensorSpec([None], dtype=tf.int32),
        tf.TensorSpec([None], dtype=tf.int32),
        tf.TensorSpec([None, None, None, None], dtype=tf.float32),
        tf.TensorSpec([None, None, None, None], dtype=tf.float32)
    ])
    def forward_encoder_chunk(
        self,
        xs: tf.Tensor,
        xs_lens: tf.Tensor,
        offset: tf.Tensor,
        required_cache_size: tf.Tensor,
        att_cache: tf.Tensor,
        cnn_cache: tf.Tensor,
    ):
        """ export savedmodel for c++ call
        """
        # TODO: batch support
        encoder_out, r_att_cache, r_cnn_cache = self.model.encoder.forward_chunk(
            xs,
            xs_lens,
            offset,
            required_cache_size,
            att_cache,
            cnn_cache,
        )
        encoder_out_lens = tf.reduce_sum(tf.cast(tf.squeeze(encoder_out_mask),
                                                 dtype=tf.int32),
                                         axis=-1)
        ctc_log_probs = self.model.ctc_dense.log_softmax(encoder_out)
        return {
            "encoder_out": encoder_out,
            "encoder_out_lens": encoder_out_lens,
            "att_cache": r_att_cache,
            "cnn_cache": r_cnn_cache,
            "ctc_activation": ctc_log_probs,
        }

    def forward_attention_decoder(self):
        """ export savedmodel for c++ call
        """
        pass

    @tf.function(input_signature=[tf.TensorSpec(None, dtype=tf.bool)])
    def metadata(self, dummy: tf.Tensor):
        """ export savedmodel for c++ call
        """

        if hasattr(self.model.decoder, 'right_decoder'):
            is_bi_decoder = tf.constant(True)
        else:
            is_bi_decoder = tf.constant(False)

        return {
            "sample_rate":
            tf.constant(self.model.encoder.embed.subsampling_rate),
            "right_context":
            tf.constant(self.model.encoder.embed.right_context),
            "sos": tf.constant(self.model.sos, dtype=tf.int32),
            "eos": tf.constant(self.model.eos, dtype=tf.int32),
            "is_bidirectional_decoder": is_bi_decoder,
        }
