"""Decoder self-attention layer definition."""
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow._api.v2.nn import dropout


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

        self.dropout_rate = dropout_rate
        self.pre_norm = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            # [size+size, size]
            self.concat_linear1 = tf.keras.layers.Dense(
                size,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )
            self.concat_linear2 = tf.keras.layers.Dense(
                size,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )

    def call(
        self,
        tgt: tf.Tensor,
        tgt_mask: tf.Tensor,
        memory: tf.Tensor,
        memory_mask: tf.Tensor,
        att_cache: Optional[tf.Tensor] = None,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute decoded features.
        Args:
            tgt (tf.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (tf.Tensor): Mask for input tensor
                (#batch, maxlen_out).
            memory (tf.Tensor): Encoded memory
                (#batch, maxlen_in, size).
            memory_mask (tf.Tensor): Encoded memory mask
                (#batch, maxlen_in).
            att_cache (tf.Tensor): Cache tensor of the KEY & VALUE
                    (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
                valid when training=False

                (#batch, maxlen_out - 1, size).
        Returns:
            tf.Tensor: Output tensor (#batch, maxlen_out, size).
            tf.Tensor: slef att ache  for output tensor
        """
        residual = tgt
        if self.pre_norm:
            tgt = self.norm1(tgt)

        tgt_q = tgt
        tgt_q_mask = tgt_mask

        # self attention
        x_att, new_att_cache = self.self_attn(
            tgt_q,
            tgt,
            tgt,
            tgt_q_mask,
            training=training,
        )
        if self.concat_after:
            tgt_concat = tf.concat([tgt_q, x_att], axis=-1)
            x = residual + self.concat_linear1(tgt_concat)
        else:
            if training:
                x = residual + tf.nn.dropout(x_att, rate=self.dropout_rate)
            else:
                x = residual + x_att
        if not self.pre_norm:
            x = self.norm1(x)

        # cross attention
        residual = x
        if self.pre_norm:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = tf.concat(
                (x,
                 self.src_attn(
                     x, memory, memory, memory_mask, training=training)[0]),
                axis=-1)
            x = residual + self.concat_linear2(x_concat)
        else:
            if training:
                x = residual + tf.nn.approx_max_k.dropout(
                    self.src_attn(
                        x, memory, memory, memory_mask, training=training)[0],
                    rate=self.dropout_rate)
            else:
                x = residual + self.src_attn(
                    x, memory, memory, memory_mask, training=False)
        if not self.pre_norm:
            x = self.norm2(x)

        residual = x
        if self.pre_norm:
            x = self.norm3(x)
        x = residual + self.feed_forward(x, training=training)
        if not self.pre_norm:
            x = self.norm3(x)

        return x, new_att_cache
