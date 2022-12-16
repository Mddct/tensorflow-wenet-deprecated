from typing import Optional
from keras.backend import update
import tensorflow as tf
from wenet.transformer.ctc import CTCDense
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import TransformerEncoder
from wenet.utils.common import (IGNORE_ID, add_sos_eos, label_smoothing_loss,
                                reverse_pad_list)


class ASRModel(tf.keras.Model):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctcdense: CTCDense,
        # for BiTransformerDecoder
        reverse_decoder: Optional[TransformerDecoder] = None,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        **kwargs,
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.smoothing = lsm_weight

        self.encoder = encoder
        if self.ctc_weight != 1:
            self.decoder = decoder
        if self.ctc_weight != 0:
            self.ctc_dense = ctcdense
        if self.reverse_weight != 0.0:
            assert self.reverse_decoder is not None
            self.reverse_decoder = reverse_decoder

    @tf.function(input_signature=[(tf.TensorSpec([None, None, None],
                                                 dtype=tf.float32),
                                   tf.TensorSpec([None], dtype=tf.int32),
                                   tf.TensorSpec([None, None], dtype=tf.int32),
                                   tf.TensorSpec([None], dtype=tf.int32))],
                 reduce_retracing=True)
    def call(
        self,
        inputs,
    ):
        """Frontend + Encoder + Decoder + Calc loss
        Args:
            inputs:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """

        speech, speech_lengths, text, text_lengths = inputs
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(
            speech,
            speech_lengths,
            training=True,
        )
        encoder_out_lens = tf.reduce_sum(tf.cast(tf.squeeze(encoder_mask,
                                                            axis=1),
                                                 dtype=text_lengths.dtype),
                                         axis=1)  # [B,]
        if self.ctc_weight != 1.0:
            decoder_out, ys_out_pad, r_decoder_out, r_ys_out_pad = self.forward_decoder(
                encoder_out, encoder_mask, text, text_lengths)
        else:
            decoder_out, ys_out_pad, r_decoder_out, r_ys_out_pad = None, None, None, None

        if self.ctc_weight != 0.0:
            # NOTE: we will use sparse tensor cuda ctc
            # but cuda ctc doesn't zero out gradients when exceeds the seq_lens
            # which makes training wrong, so turn value to zero
            # https://github.com/tensorflow/tensorflow/issues/41280
            encoder_out = self.ctc_dense(encoder_out, training=True)
            encoder_out_before = encoder_out

            @tf.custom_gradient
            def ctc_zero_out(input):

                def grad(upstream):
                    return upstream * tf.transpose(encoder_mask, [0, 2, 1])

                return tf.identity(input), grad

            encoder_out = ctc_zero_out(encoder_out_before)
            # encoder_out = encoder_out * tf.transpose(encoder_mask, [0, 2, 1])
        return (
            encoder_out,
            encoder_out_lens,
            decoder_out,
            ys_out_pad,
            r_decoder_out,
            r_ys_out_pad,
        )

    def forward_decoder(self, encoder_out: tf.Tensor, encoder_mask: tf.Tensor,
                        ys_pad: tf.Tensor, ys_pad_lens: tf.Tensor):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, ys_pad_lens, self.sos,
                                            self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        ys_pad_mask = tf.sequence_mask(ys_in_lens,
                                       maxlen=tf.shape(ys_pad)[1],
                                       dtype=encoder_mask.dtype)
        # 1. Forward decoder
        decoder_out = self.decoder(encoder_out, encoder_mask, ys_in_pad,
                                   ys_pad_mask)
        r_decoder_out = None
        r_ys_out_pad = None
        if self.reverse_weight != 0.0:
            # reverse the seq, used for right to left decoder
            r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens)
            r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, ys_pad_lens,
                                                    self.sos, self.eos,
                                                    self.ignore_id)
            r_decoder_out = self.reverse_decoder(encoder_out, encoder_mask,
                                                 r_ys_in_pad, ys_pad_mask)

        return decoder_out, ys_out_pad, r_decoder_out, r_ys_out_pad

    @tf.function
    def compute_loss(
        self,
        encoder_logits,
        encoder_logits_lens,
        encoder_labels,
        encoder_labels_lens,
        decoder_logits,
        decoder_out_pad,
        r_decoder_logits,
        r_decoder_out_pad,
    ):
        loss_ctc = None
        # TODO: move label convert to data pipeline
        # NOTE: sparse label, tf will choose cuda ctc which faster than dense even on gpu
        if self.ctc_weight != 0.0:
            labels_sparse = tf.sparse.from_dense(encoder_labels)
            loss_ctc = tf.nn.ctc_loss(
                labels_sparse,
                encoder_logits,
                None,
                encoder_logits_lens,
                logits_time_major=False,
                blank_index=0,
            )
        loss_att = None
        if self.ctc_weight != 1.0:
            loss_att = label_smoothing_loss(decoder_out_pad, decoder_logits,
                                            self.vocab_size, self.ignore_id,
                                            self.smoothing)
            if self.reverse_weight > 0.0:
                r_loss_att = label_smoothing_loss(
                    r_decoder_out_pad,
                    r_decoder_logits,
                    self.vocab_size,
                    self.ignore_id,
                    self.smoothing,
                )
                loss_att = loss_att * (
                    1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        loss = None
        if self.ctc_weight == 0.0:
            loss = loss_att
        elif self.ctc_weight == 1.0:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
        return {"loss": loss, "loss_ctc": loss_ctc, "loss_att": loss_att}
