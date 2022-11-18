"""Decoder definition."""
from typing import List, Optional, Tuple

import tensorflow as tf
from typeguard import check_argument_types
from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.decoder_layer import DecoderLayer
from wenet.transformer.embedding import (EmbeddingSharedWeights,
                                         PositionalEncoding)
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.utils.mask import subsequent_mask


class TransformerDecoder(tf.keras.layers.Layer):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """

    def __init__(
            self,
            vocab_size: int,
            encoder_output_size: int,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 6,
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            self_attention_dropout_rate: float = 0.0,
            src_attention_dropout_rate: float = 0.0,
            input_layer: str = "embed",
            use_output_layer: bool = True,
            output_layer_share_weights: bool = False,
            normalize_before: bool = True,
            concat_after: bool = False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
            bias_regularizer=tf.keras.regularizers.l2(1e-6),
    ):
        assert check_argument_types()
        super().__init__()
        attention_dim = encoder_output_size

        self.d_model = attention_dim
        if input_layer == "embed":
            self.look_up = EmbeddingSharedWeights(vocab_size, self.d_model)
            self.position = PositionalEncoding(attention_dim,
                                               positional_dropout_rate)
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.after_norm = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            epsilon=1e-6,
        )

        self.use_output_layer = use_output_layer

        # [attention_dim vocab_size]
        self.output_layer_share_weights = output_layer_share_weights
        if output_layer_share_weights:
            self.output_layer = self.look_up
        else:
            self.output_layer = tf.keras.layers.Dense(
                vocab_size,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )

        self.num_blocks = num_blocks
        self.decoders = [
            DecoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim,
                                     self_attention_dropout_rate),
                MultiHeadedAttention(attention_heads, attention_dim,
                                     src_attention_dropout_rate),
                PositionwiseFeedForward(attention_dim, linear_units,
                                        dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(self.num_blocks)
        ]

    def call(
        self,
        memory: tf.Tensor,
        memory_mask: tf.Tensor,
        ys_in_pad: tf.Tensor,
        ys_in_lens: tf.Tensor,
        r_ys_in_pad: tf.Tensor,
        reverse_weight: float = 0.0,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: not used in transformer decoder, in order to unify api
                with bidirectional decoder
            reverse_weight: not used in transformer decoder, in order to unify
                api with bidirectional decode
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                tf.tensor(0.0), in order to unify api with bidirectional decoder
        """
        tgt = ys_in_pad
        tgt_shape = tf.shape(tgt)
        maxlen = tgt_shape[1]

        # tgt_mask: (B, 1, L)
        tgt_mask = tf.expand_dims(tf.sequence_mask(ys_in_lens, maxlen), axis=1)

        # m: (1, L, L)
        m = tf.expand_dims(subsequent_mask(maxlen), axis=0)
        # tgt_mask: (B, L, L)
        tgt_mask = tgt_mask & m

        # x, _ = self.embed(tgt)
        fake_offset = tf.zeros([tgt_shape[0]], dtype=tgt.dtype)
        x = self.look_up(tgt)
        x, _ = self.position([x, fake_offset], training=training)

        # memory_mask = tf.transpose(memory_mask, [0, 2, 1])  # [B,1,T]
        for layer in self.decoders:
            x, _ = layer(x, tgt_mask, memory, memory_mask, training=training)
        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            if self.output_layer_share_weights:
                x = self.output_layer(x, mode="linear")
            else:
                x = self.output_layer(x)
        # olens = tf.reduce_sum(tgt_mask, axis=1)
        return x, tf.constant(0.0, dtype=x.dtype)

    def _get_symbols_to_logits_fn(self, max_decode_length, training):
        """Returns a decoding function that calculates logits of the next tokens."""
        # TODO
        pass


class BiTransformerDecoder(tf.keras.layers.Layer):
    """Base class of Transfomer decoder module.
    Args:
        vocab_size: output dim
        encoder_output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the hidden units number of position-wise feedforward
        num_blocks: the number of decoder blocks
        r_num_blocks: the number of right to left decoder blocks
        dropout_rate: dropout rate
        self_attention_dropout_rate: dropout rate for attention
        input_layer: input layer type
        use_output_layer: whether to use output layer
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before:
            True: use layer_norm before each sub-block of a layer.
            False: use layer_norm after each sub-block of a layer.
        concat_after: whether to concat attention layer's input and output
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        r_num_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        self_attention_dropout_rate: float = 0.0,
        src_attention_dropout_rate: float = 0.0,
        input_layer: str = "embed",
        use_output_layer: bool = True,
        normalize_before: bool = True,
        concat_after: bool = False,
    ):

        assert check_argument_types()
        super().__init__()
        self.left_decoder = TransformerDecoder(
            vocab_size, encoder_output_size, attention_heads, linear_units,
            num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            input_layer, use_output_layer, normalize_before, concat_after)

        self.right_decoder = TransformerDecoder(
            vocab_size, encoder_output_size, attention_heads, linear_units,
            r_num_blocks, dropout_rate, positional_dropout_rate,
            self_attention_dropout_rate, src_attention_dropout_rate,
            input_layer, use_output_layer, normalize_before, concat_after)

    def call(
        self,
        memory: tf.Tensor,
        memory_mask: tf.Tensor,
        ys_in_pad: tf.Tensor,
        ys_in_lens: tf.Tensor,
        r_ys_in_pad: tf.Tensor,
        reverse_weight: float = 0.0,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Forward decoder.
        Args:
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            memory_mask: encoder memory mask, (batch, 1, maxlen_in)
            ys_in_pad: padded input token ids, int64 (batch, maxlen_out)
            ys_in_lens: input lengths of this batch (batch)
            r_ys_in_pad: padded input token ids, int64 (batch, maxlen_out),
                used for right to left decoder
            reverse_weight: used for right to left decoder
        Returns:
            (tuple): tuple containing:
                x: decoded token score before softmax (batch, maxlen_out,
                    vocab_size) if use_output_layer is True,
                r_x: x: decoded token score (right to left decoder)
                    before softmax (batch, maxlen_out, vocab_size)
                    if use_output_layer is True,
        """
        l_x, _, = self.left_decoder(memory,
                                    memory_mask,
                                    ys_in_pad,
                                    ys_in_lens,
                                    training=training)
        r_x = tf.constant(0.0, l_x.dtype)
        if reverse_weight > 0.0:
            r_x, _ = self.right_decoder(memory,
                                        memory_mask,
                                        r_ys_in_pad,
                                        ys_in_lens,
                                        training=training)
        return l_x, r_x
