import tensorflow as tf
from wenet.utils.common import reverse_pad_list, add_sos_eos


class Decoder(tf.Module):

    def __init__(
        self,
        model: tf.keras.Model,
        chunk_size=-1,
        num_decoding_left_chunks=-1,
        num_layers=12,
        conv_cacche=True,
    ):
        """ Decodef for python and runtime inference
        """
        super(Decoder, self).__init__()
        self.model = model
        self.chunk_size = chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks
        self.num_layers = num_layers
        self.conv_cache = conv_cacche  # for conforemr, false transformer

    def forward_encoder(
        self,
        xs: tf.Tensor,
        xs_lens: tf.Tensor,
    ):
        """ python inference"""
        encoder_out, encoder_out_mask = self.model.encoder(
            xs,
            xs_lens,
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
        beam_scores: tf.Tensor,
        beam_weight: tf.Tensor,
        reverse_weight: tf.Tensor,
        encoder_out: tf.Tensor,
        encoder_out_mask: tf.Tensor,
    ):
        """ attention rescoring

        Args:
          hyps (tf.Tensor): [B, beam, max_length]
          hyps_lens (tf.Tensor): [B, beam]
          beam_scores (tf.Tensor): [B, beam]
          encoder_out (tf.Tensor): [B, T, dim]
          encoder_out_mask (tf.Tensor): [B, 1, T]

        Returns:
          final score: [B, beam]
          rescore scores: [B, beam]
          reverse rescore scores: [B, beam]
        """
        batch_size, beam = tf.shape(hyps)[0], tf.shape(hyps)[1]
        hyps = tf.reshape(hyps,
                          [batch_size * beam, -1])  # [B*beam, max_length]
        hyps_lens = tf.reshape(hyps, [-1])  # [B*beam]
        encoder_out = tf.tile(encoder_out, [beam, 1, 1])  # [B*beam, T, dim]
        encoder_out_mask = tf.tile(encoder_out_mask,
                                   [beam, 1, 1])  # [B*beam, 1, T]

        sos_hyps, hyps_eos = add_sos_eos(hyps, hyps_lens, self.model.sos,
                                         self.model.eos, self.model.ignore_id)
        decoder_out = self.model.decoder(encoder_out,
                                         encoder_out_mask,
                                         sos_hyps,
                                         hyps_lens + 1,
                                         training=False)
        decoder_out_scores = tf.nn.log_softmax(decoder_out,
                                               axis=-1)  #[B*beam, L, vocab]
        decoder_out_scores = tf.gather(
            params=decoder_out_scores,
            indices=hyps_eos,
            axis=2,
            batch_dims=2,
        )  # [B*beam, L]
        # TODO: mask should return by decoder not here
        mask = tf.sequence_mask(hyps_lens + 1,
                                dtype=decoder_out_scores,
                                maxlen=tf.shape(decoder_out_scores)[-1])
        decoder_out_scores = mask * decoder_out_scores
        decoder_out_scores = tf.reduce_sum(tf.reshape(decoder_out_scores,
                                                      [batch_size, beam, -1]),
                                           axis=-1,
                                           keepdims=False)
        r_decoder_out = 0
        if hasattr(self.model, 'reverse_decoder'):
            r_hyps = reverse_pad_list(hyps, hyps_lens)
            sos_r_hyps, r_hyps_eos = add_sos_eos(r_hyps, hyps_lens,
                                                 self.model.sos,
                                                 self.model.eos,
                                                 self.model.ignore_id)
            r_decoder_out = self.model.reverse_decoder(encoder_out,
                                                       encoder_out_mask,
                                                       sos_r_hyps,
                                                       hyps_lens + 1,
                                                       training=False)

            r_decoder_out_scores = tf.nn.log_softmax(
                r_decoder_out, axis=-1)  #{B*beam, L, vocab]}
            r_decoder_out_scores = tf.gather(
                params=r_decoder_out_scores,
                indices=r_hyps_eos,
                axis=2,
                batch_dims=2,
            )  # [B*beam, L]
            r_decoder_out_scores = mask * r_decoder_out_scores
            r_decoder_out_scores = tf.reduce_sum(tf.reshape(
                decoder_out_scores, [batch_size, beam, -1]),
                                                 axis=-1,
                                                 keepdims=False)

        final_scores = beam_scores * beam_weight + (1 - beam_weight) * (
            (1 - reverse_weight) * decoder_out_scores +
            reverse_weight * r_decoder_out_scores)
        return {
            "rescore_scores": decoder_out_scores,
            "reverse_rescore_scores": r_decoder_out_scores,
            "final_scores": final_scores,
        }

    def _get_symbols_to_logits_fn(self, max_decode_length):
        """Returns a decoding function that calculates logits of the next tokens.
           python inference
        """
        decoder = self.model.decoder
        return decoder._get_symbos_to_logits_fn(max_decode_length,
                                                training=False)

    @tf.function(input_signature=[
        tf.TensorSpec([None, None, None], dtype=tf.float32),
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
        att_cache_mask: tf.Tensor,
        cnn_cache: tf.Tensor,
    ):
        """ export savedmodel for c++ call
        """
        key_cache, value_cache = tf.split(att_cache,
                                          num_or_size_splits=2,
                                          axis=-1)
        att_cache_dict = {
            layer: {
                "k": key_cache[:, layer, :, :, :],
                "v": value_cache[:, layer, :, :, :],
            }
            for layer in range(self.num_layers)
        }
        # cnn_cache_dict = {'conv': cnn_cache}
        cnn_cache_dict = {
            layer: {
                "conv": cnn_cache[:, layer, :, :]
            }
            for layer in range(self.num_layers)
        }
        att_cache_mask_dict = {
            "mask": att_cache_mask,
        }
        encoder_out, encoder_out_mask = self.model.encoder.forward_chunk(
            xs,
            xs_lens,
            offset,
            required_cache_size,
            att_cache_dict,
            att_cache_mask_dict,
            cnn_cache_dict,
        )
        encoder_out_lens = tf.reduce_sum(tf.cast(tf.squeeze(encoder_out_mask),
                                                 dtype=tf.int32),
                                         axis=-1)
        ctc_log_probs = self.model.ctc_dense.log_softmax(encoder_out)

        # combine layer cache to a tensor
        r_att_cache = []
        r_cnn_cache = []
        for layer in range(self.num_layers):
            k, v = att_cache_dict[layer]['k'], att_cache_dict[layer]['v']
            r_att_cache.append(
                tf.expand_dims(tf.concat([k, v], axis=-1), axis=1))

            if self.conv_cache:
                r_cnn_cache.append(
                    tf.expand_dims(cnn_cache_dict[layer]['conv'], axis=1))

        new_att_cache = tf.concat(r_att_cache, axis=1)
        if self.conv_cache:
            new_cnn_cache = tf.concat(r_cnn_cache, axis=1)
        else:
            new_cnn_cache = cnn_cache
        new_att_cache_mask = att_cache_mask_dict['mask']

        return {
            "encoder_out": encoder_out,
            "encoder_out_lens": encoder_out_lens,
            "att_cache": new_att_cache,
            "att_cache_mask": new_att_cache_mask,
            "cnn_cache": new_cnn_cache,
            "ctc_activation": ctc_log_probs,
        }

    @tf.function(input_signature=[
        tf.TensorSpec([None, None, None], dtype=tf.float32),
        tf.TensorSpec([None], dtype=tf.int32),
        tf.TensorSpec([None, None], dtype=tf.int32),
        tf.TensorSpec((), dtype=tf.int32),
        tf.TensorSpec((), dtype=tf.int32),
        tf.TensorSpec([None, None, None], dtype=tf.float32),
        tf.TensorSpec([None, None], dtype=tf.float32)
    ])
    def forward_attention_decoder(
        self,
        hyps: tf.Tensor,
        hyps_lens: tf.Tensor,
        beam_scores: tf.Tensor,
        beam_weight: tf.Tensor,
        reverse_weight: tf.Tensor,
        encoder_out: tf.Tensor,
        encoder_out_mask: tf.Tensor,
    ):
        """ export savedmodel for c++ call
        """

        return self.attention_rescoring(
            hyps,
            hyps_lens,
            beam_scores,
            beam_weight,
            reverse_weight,
            encoder_out,
            encoder_out_mask,
        )

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
