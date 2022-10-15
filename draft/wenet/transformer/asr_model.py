from typing import Dict, Optional

import tensorflow as tf
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import IGNORE_ID, add_sos_eos, reverse_pad_list


class ASRModel(tf.keras.Model):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
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

        self.encoder = encoder
        if self.ctc_weight != 1:
            self.decoder = decoder
            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
            )
        if self.ctc_weight != 0:
            self.ctc = ctc

    def call(
        self,
        speech: tf.Tensor,
        speech_lengths: tf.Tensor,
        text: tf.Tensor,
        text_lengths: tf.Tensor,
    ) -> Dict[str, Optional[tf.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        # 1. Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths)
        encoder_out_lens = tf.reduce_sum(tf.cast(tf.squeeze(encoder_mask,
                                                            axis=1),
                                                 dtype=text_lengths.dtype),
                                         axis=1)  # [B,]
        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att = self._calc_att_loss(encoder_out, encoder_mask, text,
                                           text_lengths)
        else:
            loss_att = None

        # 2b. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
        return {"loss": loss, "loss_att": loss_att, "loss_ctc": loss_ctc}

    def _calc_att_loss(
        self,
        encoder_out: tf.Tensor,
        encoder_mask: tf.Tensor,
        ys_pad: tf.Tensor,
        ys_pad_lens: tf.Tensor,
    ) -> tf.Tensor:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, ys_pad_lens, self.sos,
                                            self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, self.ignore_id)
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, ys_pad_lens,
                                                self.sos, self.eos,
                                                self.ignore_id)
        # 1. Forward decoder
        decoder_out, r_decoder_out = self.decoder(encoder_out, encoder_mask,
                                                  ys_in_pad, ys_in_lens,
                                                  r_ys_in_pad,
                                                  self.reverse_weight)
        # 2. Compute attention loss
        loss_att = self.criterion_att(ys_out_pad, decoder_out)
        r_loss_att = 0.0
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (
            1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        return loss_att
