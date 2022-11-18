"""Encoder self-attention layer definition."""

from typing import Optional, Tuple
from keras.backend_config import epsilon

import tensorflow as tf


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (tf.keras.layers.Layer): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (tf.keras.layers.Layer): Feed-forward module instance.
            `PositionwiseFeedForward`, instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: to use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """

    def __init__(self,
                 size: int,
                 self_attn: tf.keras.layers.Layer,
                 feed_forward: tf.keras.layers.Layer,
                 dropout_rate: float,
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 bias_regularizer="l2",
                 kernel_regularizer="l2",
                 trainable=True,
                 name="TransformerEncoderLayer",
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            epsilon=1e-6,
        )
        self.norm2 = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            epsilon=1e-6,
        )

        self.dropout_rate = dropout_rate

        self.size = size
        self.pre_nrom = normalize_before
        self.concat_after = concat_after
        if concat_after:
            self.concat_linear = tf.keras.layers.Dense(
                size,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )

    def call(
        self,
        inputs: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute encoded features.
        Args:
            inputs:
                x (tf.Tensor): (#batch, time, size)
                att_cache (tf.Tensor): Cache tensor of the KEY & VALUE
                    (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
                cnn_cache (tf.Tensor): Convolution cache in conformer layer
                    (#batch=1, size, cache_t2), not used here, it's for interface
                    compatibility to ConformerEncoderLayer.

            mask (tf.Tensor): Mask tensor for the input (#batch, time，time),

        Returns:
            tf.Tensor: Output tensor (#batch, time, size).
            tf.Tensor: Mask tensor (#batch, time, time).
            tf.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            tf.Tensor: cnn_cahce tensor (#batch=1, size, cache_t2).
        """
        x, _, _, att_cache, _ = inputs
        residual = x
        if self.pre_nrom:
            x = self.norm1(x)

        x_att, new_att_cache = self.self_attn(x,
                                              x,
                                              x,
                                              mask,
                                              cache=att_cache,
                                              training=training)
        if self.concat_after:
            x_concat = tf.concat((x, x_att), axis=-1)
            if training:
                x = residual + tf.nn.dropout(self.concat_linear(x_concat),
                                             rate=self.dropout_rate)
            else:
                x = residual + self.concat_linear(x_concat)
        else:
            if training:
                x = residual + tf.nn.dropout(x_att, rate=self.dropout_rate)
            else:
                x = residual + x_att

        if not self.pre_nrom:
            x = self.norm1(x)

        residual = x
        if self.pre_nrom:
            x = self.norm2(x)
        x = residual + self.feed_forward(x, training=training)
        if not self.pre_nrom:
            x = self.norm2(x)

        fake_cnn_cache = tf.zeros((0, 0, 0), dtype=x.dtype)
        return x, mask, new_att_cache, fake_cnn_cache


class ConformerEncoderLayer(tf.keras.layers.Layer):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (tf.keras.layers.Layer): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (tf.keras.layers.Layer): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (tf.keras.layers.Layer): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (tf.keras.layers.Layer): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """

    def __init__(self,
                 size: int,
                 self_attn: tf.keras.layers.Layer,
                 feed_forward: Optional[tf.keras.layers.Layer] = None,
                 feed_forward_macaron: Optional[tf.keras.layers.Layer] = None,
                 conv_module: Optional[tf.keras.layers.Layer] = None,
                 dropout_rate: float = 0.1,
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 bias_regularizer="l2",
                 kernel_regularizer="l2",
                 trainable=True,
                 name="ConformerEncoderLayer",
                 dtype=None,
                 dynamic=False,
                 **kwargs):
        """Construct an EncoderLayer object."""
        super(ConformerEncoderLayer, self).__init__(trainable, name, dtype,
                                                    dynamic, **kwargs)
        assert conv_module is not None
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module

        # self.norm_ff = nn.LayerNorm(size, eps=1e-5)  # for the FNN module
        # self.norm_mha = nn.LayerNorm(size, eps=1e-5)  # for the MHA module
        self.norm_ff = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            epsilon=1e-6,
        )
        self.norm_mha = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            ellipsis=1e-6,
        )

        self.macaron = True if feed_forward_macaron is not None else False
        if self.macaron:
            # self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
            self.norm_ff_macaron = tf.keras.layers.LayerNormalization(
                gamma_regularizer=kernel_regularizer,
                beta_regularizer=bias_regularizer,
                epsilon=1e-6,
            )
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        # self.norm_conv = nn.LayerNorm(size, eps=1e-5)  # for the CNN module
        # self.norm_final = nn.LayerNorm(
        #     size, eps=1e-5)  # for the final output of the block
        self.norm_conv = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            epsilon=1e-6,
        )
        self.norm_final = tf.keras.layers.LayerNormalization(
            gamma_regularizer=kernel_regularizer,
            beta_regularizer=bias_regularizer,
            epsilon=1e-6,
        )
        # self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate

        self.size = size
        self.pre_norm = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = tf.keras.layers.Dense(
                size,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            )

    def call(
        self,
        inputs: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Compute encoded features.
        Args:
            inputs:
                x (tf.Tensor): (#batch, time, size)
                pos_emb (tf.Tensor): positional encoding, must not be None
                    for ConformerEncoderLayer.
                mask_pad (tf.Tensor): batch padding mask used for conv module.
                    (#batch, time,1 )
                att_cache (tf.Tensor): Cache tensor of the KEY & VALUE
                    (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
                cnn_cache (tf.Tensor): Convolution cache in conformer layer
                    (#batch=1, size, cache_t2)
            mask (tf.Tensor): Mask tensor for the input (#batch, time，time),

        Returns:
            tf.Tensor: Output tensor (#batch, time, size).
            tf.Tensor: Mask tensor (#batch, time, time).
            tf.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
                valid when training=False
            tf.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
                valid when training=False
        """

        x, pos_emb, mask_pad, att_cache, cnn_cache = inputs
        # whether to use macaron style
        if self.macaron:
            residual = x
            if self.pre_norm:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.feed_forward_macaron(
                x, training=training)
            if not self.pre_norm:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.pre_norm:
            x = self.norm_mha(x)

        x_att, new_att_cache = self.self_attn(x,
                                              x,
                                              x,
                                              mask,
                                              pos_emb,
                                              att_cache,
                                              training=training)
        if self.concat_after:
            x_concat = tf.concat((x, x_att), axis=-1)
            if training:
                x = residual + tf.nn.dropout(self.concat_linear(x_concat),
                                             rate=self.dropout_rate)
            else:
                x = residual + self.concat_linear(x_concat)
        else:
            if training:
                x = residual + tf.nn.dropout(x_att, rate=self.dropout_rate)
            else:
                x = residual + x_att
        if not self.pre_norm:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        residual = x
        if self.pre_norm:
            x = self.norm_conv(x)
        x, new_cnn_cache = self.conv_module(x,
                                            mask_pad,
                                            cnn_cache,
                                            training=training)
        x = residual + x
        if not self.pre_norm:
            x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.pre_norm:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.feed_forward(x, training=training)
        if not self.pre_norm:
            x = self.norm_ff(x)

        x = self.norm_final(x)
        return x, mask, new_att_cache, new_cnn_cache
