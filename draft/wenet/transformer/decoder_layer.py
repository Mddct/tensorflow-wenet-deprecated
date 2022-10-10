"""Decoder self-attention layer definition."""
from typing import Optional, Tuple

import tensorflow as tf


class DecoderLayer(tf.keras.layers.Layer):
    """Single decoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (tf.keras.layers.Layer): Self-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        src_attn (tf.keras.layers.Layer): Inter-attention module instance.
            `MultiHeadedAttention` instance can be used as the argument.
        feed_forward (tf.keras.layers.Layer): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's inpu
            and output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """

    def __init__(
            self,
            size: int,
            self_attn: tf.keras.layers.Layer,
            src_attn: tf.keras.layers.Layer,
            feed_forward: tf.keras.layers.Layer,
            dropout_rate: float,
            normalize_before: bool = True,
            concat_after: bool = False,
            bias_regularizer=tf.keras.regularizers.l2(1e-6),
            kernel_regularizer=tf.keras.regularizers.l2(1e-6),
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            epsilon=1e-5,
        )
        self.norm2 = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            epsilon=1e-5,
        )

        self.norm3 = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            epsilon=1e-5,
        )

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            # [size+size, size]
            self.concat_linear1 = tf.keras.layers.Dense(size)
            self.concat_linear2 = tf.keras.layers.Dense(size)

    def call(
        self,
        tgt: tf.Tensor,
        tgt_mask: tf.Tensor,
        memory: tf.Tensor,
        memory_mask: tf.Tensor,
        cache: Optional[tf.Tensor] = None,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute decoded features.
        Args:
            tgt (tf.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (tf.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (tf.Tensor): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (tf.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            cache (tf.Tensor): cached tensors.
                (#batch, maxlen_out - 1, size).
        Returns:
            tf.Tensor: Output tensor (#batch, maxlen_out, size).
            tf.Tensor: Mask for output tensor (#batch, maxlen_out).
            tf.Tensor: Encoded memory (#batch, maxlen_in, size).
            tf.Tensor: Encoded memory mask (#batch, maxlen_in).
        """
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            # compute only the last frame query keeping dim: max_time_out -> 1
            # assert cache.shape == (
            #     tgt.shape[0],
            #     tgt.shape[1] - 1,
            #     self.size,
            # ), "{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]

        if self.concat_after:
            tgt_concat = tf.concat(
                (tgt_q,
                 self.self_attn(tgt_q, tgt, tgt, tgt_q_mask,
                                training=training)[0]),
                axis=-1)
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(
                self.self_attn(tgt_q, tgt, tgt, tgt_q_mask,
                               training=training)[0])
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = tf.concat(
                (x,
                 self.src_attn(
                     x, memory, memory, memory_mask, training=training)[0]),
                dim=-1)
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(
                x, memory, memory, memory_mask, training=training)[0],
                                        training=training)
        if not self.normalize_before:
            x = self.norm2(x)

        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x, training=training),
                                    training=training)
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = tf.concat([cache, x], axis=1)

        return x, tgt_mask, memory, memory_mask
