"""ConvolutionModule definition."""

from typing import Optional, Tuple

import tensorflow as tf
from wenet.transformer.activations import ActivationLayer


class ConvolutionModule(tf.keras.layers.Layer):
    """ConvolutionModule in Conformer model."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        activation: str = 'swish',
        norm: str = "batch_norm",
        dropout_rate: float = 0.1,
        causal: bool = False,
        bias: bool = True,
        bias_regularizer="l2",
        kernel_regularizer="l2",
        name="conv_module",
        **kwargs,
    ):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super(ConvolutionModule, self).__init__(name=name, **kwargs)

        self.pointwise_conv1 = tf.keras.layers.Conv1D(
            2 * channels,
            kernel_size=1,
            strides=1,
            padding='valid',
            name=f"{name}_pw_conv_1",
            use_bias=bias,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            # input_shape=[None, channels],
        )
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 'valid'
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = 'same'
            # padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = tf.keras.layers.Conv1D(
            channels,
            kernel_size=kernel_size,
            strides=1,
            padding=padding,  # (kenel_size-1)//2 underhood
            groups=channels,
            use_bias=bias,
            bias_regularizer=bias_regularizer,
            kernel_regularizer=kernel_regularizer,
            name=f"{name}_dw_conv",
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.norm = tf.keras.layers.BatchNormalization(
                name=f"{name}_bn",
                gamma_regularizer=kernel_regularizer,
                beta_regularizer=bias_regularizer,
                axis=-1,  # channels last
                epsilon=1e-6,
            )
        else:
            self.norm = tf.keras.layers.LayerNormalization(
                name=f"{name}_ln",
                gamma_regularizer=kernel_regularizer,
                beta_regularizer=bias_regularizer,
                epsilon=1e-6,
            )

        self.pointwise_conv2 = tf.keras.layers.Conv1D(
            channels,
            kernel_size=1,
            strides=1,
            padding='valid',  # no need padding
            use_bias=bias,
            name=f"{name}_pw_conv_2",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.glu = ActivationLayer('glu')
        self.activation = ActivationLayer(activation)
        self.dropout = tf.keras.layers.Dropout(dropout_rate,
                                               name=f"{name}_dropout")

    def call(self,
             inputs,
             mask=None,
             cache=None,
             training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute convolution module.
        Args:
            inputs:
                x (tf.Tensor): Input tensor (#batch, time, channels).
                mask_pad (tf.Tensor): used for batch padding (#batch, time, 1),
                cache (tf.Tensor Optional): left context cache, it is only
                    used in causal convolution (#batch, cache_t, channels),
                    only valid when training == false
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension

        x = inputs
        if self.lorder > 0:
            if training:  # cache_t == 0
                x = tf.pad(x, ([0, 0], [self.lorder, 0], [0, 0]))  # pad zero
            else:
                x = tf.concat((cache, x), axis=1)
            # assert (x.size(2) > self.lorder)
            new_cache = x[:, -self.lorder:, :]
        else:
            # mask batch padding
            # x = tf.where(mask_pad, x, 0.0)
            x = x * tf.cast(mask, dtype=x.dtype)
            new_cache = tf.zeros([0, 0, 0], dtype=x.dtype)

        # GLU mechanism
        x = self.pointwise_conv1(x,
                                 training=training)  # (batch, time, 2*channel)
        x = self.glu(x)  # (batch,  time,  channel)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x, training=training)
        x = self.activation(self.norm(x, training=training))
        x = self.pointwise_conv2(x, training=training)
        x = self.dropout(x, training=training)

        # mask batch padding
        if self.lorder == 0:
            x = x * tf.cast(mask, dtype=x.dtype)
        return x, new_cache
