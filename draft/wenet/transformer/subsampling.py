"""Subsampling layer definition."""

from typing import Tuple, Union

import tensorflow as tf
from wenet.transformer.embedding import NoPositionalEncoding, PositionalEncoding, RelPositionalEncoding
from wenet.transformer.activations import ActivationLayer


class BaseSubsampling(tf.keras.layers.Layer):

    def __init__(self, dropout_rate, **kwargs):
        super(BaseSubsampling, self).__init__(**kwargs)
        self.right_context = 0
        self.subsampling_rate = 1
        self.dropout_rate = dropout_rate

    def position_encoding(self,
                          offset: tf.Tensor,
                          size: tf.Tensor,
                          apply_dropout: bool = False,
                          training: bool = True) -> tf.Tensor:
        return self.pos_enc.position_encoding(offset, size, apply_dropout,
                                              training)

    def compute_mask_length(self,
                            length,
                            kernel_size: int = 3,
                            stride: int = 2,
                            dilation: int = 1,
                            padding: int = 0):
        """
        Args:
           length: [B], tf.Tensor
        """

        # return tf.math.floordiv((length + 2 * padding - dilation *
        # (kernel_size - 1) - 1), stride) + 1
        return (length + 2 * padding - dilation *
                (kernel_size - 1) - 1) // stride + 1


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
    """

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc_class: Union[PositionalEncoding, RelPositionalEncoding,
                             NoPositionalEncoding],
        bias_regularizer="l2",
        kernel_regularizer="l2",
        **kwargs,
    ):
        """Construct an linear object."""
        super(LinearNoSubsampling, self).__init__(dropout_rate=dropout_rate,
                                                  **kwargs)
        self.out = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[None, idim, 1]),
            tf.keras.layers.Dense(
                odim,
                bias_regularizer=bias_regularizer,
                kernel_regularizer=kernel_regularizer,
            ),
            tf.keras.layers.LayerNormalization(
                epsilon=1e-6,
                beta_regularizer=kernel_regularizer,
                gamma_regularizer=kernel_regularizer,
            ),
        ])

        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Input x.
        Args:
            inputs:
                x (tf.Tensor): Input tensor (#batch, time, idim).
                offset(tf.Tensor):
            mask (tf.Tensor): Input mask (#batch, time, 1).
        Returns:
            tf.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            tf.Tensor: pos embedding
        """
        x, offset = inputs
        x = self.out(x, training=training)
        if training:
            x = tf.nn.dropout(x, self.dropout_rate)
        x, pos_emb = self.pos_enc((x, offset), training=training)
        return x, pos_emb

    def get_mask(self, lens, mask=None):
        return tf.sequence_mask(lens)


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc_class: Union[PositionalEncoding, RelPositionalEncoding,
                             NoPositionalEncoding],
        bias_regularizer="l2",
        kernel_regularizer="l2",
        **kwargs,
    ):
        """Construct an Conv2dSubsampling4 object."""
        super(Conv2dSubsampling4, self).__init__(dropout_rate=dropout_rate,
                                                 **kwargs)
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[None, idim, 1]),
            tf.keras.layers.Conv2D(
                filters=odim,
                strides=2,
                kernel_size=3,
                bias_regularizer=bias_regularizer,
                kernel_regularizer=kernel_regularizer,
            ),
            ActivationLayer('relu'),
            tf.keras.layers.Conv2D(
                filters=odim,
                strides=2,
                kernel_size=3,
                bias_regularizer=bias_regularizer,
                kernel_regularizer=kernel_regularizer,
            ),
            ActivationLayer('relu'),
        ])

        # input dim should  == odim * (((idim - 1) // 2 - 1) // 2)
        self.out = tf.keras.layers.Dense(
            odim,
            bias_regularizer=bias_regularizer,
            kernel_regularizer=kernel_regularizer,
        )
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Subsample x.
        Args:
            inputs:
                x (tf.Tensor): Input tensor (#batch, time, idim).
                x_mask (tf.Tensor): Input mask (#batch, time, 1).
        Returns:
            tf.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            tf.Tensor: Subsampled mask (#batch, time', 1),
                where time' = time // 4.
            tf.Tensor: positional encoding
        """
        x, offset = inputs
        x = tf.expand_dims(x, axis=3)  # (b, t, f, 1)
        # x = self.conv1(x)  # (b, t', f', 1)
        # x = tf.nn.relu(x)
        # x = self.conv2(x)  # (b, t'', f'', odim)
        x = self.conv(x)
        x_shape = tf.shape(x)
        b, t, f, c = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        x = self.out(tf.reshape(x, [b, t, f * c]))
        if training:
            x = tf.nn.dropout(x, self.dropout_rate)
        x, pos_emb = self.pos_enc((x, offset), training=training)
        return x, pos_emb

    def get_mask(self, lens, mask=None):
        lens = self.compute_mask_length(lens, kernel_size=3, stride=2)
        lens = self.compute_mask_length(lens, kernel_size=3, stride=2)
        mask = tf.sequence_mask(lens)
        return mask


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (tf.keras.layers.Layer): Custom position encoding layer.
    """

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc_class: Union[PositionalEncoding, RelPositionalEncoding,
                             NoPositionalEncoding],
        bias_regularizer="l2",
        kernel_regularizer="l2",
        **kwargs,
    ):
        """Construct an Conv2dSubsampling6 object."""
        super(Conv2dSubsampling6, self).__init__(**kwargs)
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, idim, 1)),
            tf.keras.layers.Conv2D(
                odim,
                3,
                2,
                bias_regularizer=bias_regularizer,
                kernel_regularizer=kernel_regularizer,
            ),
            ActivationLayer('relu'),
            tf.keras.layers.Conv2D(
                odim,
                5,
                3,
                bias_regularizer=bias_regularizer,
                kernel_regularizer=kernel_regularizer,
            ),
            ActivationLayer('relu')
        ])
        # input shape:  odim * ((idim-1)//2-2//3)
        self.out = tf.keras.layers.Dense(
            odim,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.pos_enc = pos_enc_class
        # 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.subsampling_rate = 6
        self.right_context = 10

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Subsample x.
        Args:
            inputs:
                x (tf.Tensor): Input tensor (#batch, time, idim).
                offset (tf.Tensor)
            x_mask (tf.Tensor): Input mask (#batch, time, 1).
        Returns:
            tf.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            tf.Tensor: Subsampled mask (#batch, time', 1),
                where time' = time // 6.
            tf.Tensor: positional encoding
        """
        x, offset = inputs
        x = tf.expand_dims(x, axis=3)  # (b, t, f, 1)
        # x = self.conv1(x)  # (b, t', f', 1)
        # x = tf.nn.relu(x)
        # x = self.conv2(x)  # (b, t'', f'', odim)
        x = self.conv(x)
        x_shape = tf.shape(x)
        b, t, f, c = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        x = self.out(tf.reshape(x, [b, t, f * c]))
        if training:
            x = tf.nn.dropout(x, self.dropout_rate)
        x, pos_emb = self.pos_enc((x, offset), training=training)
        # return x, pos_emb, mask[:, :-2:2, :][:, :-4:3, :]
        return x, pos_emb

    def get_mask(self, lens, mask=None):
        lens = self.compute_mask_length(lens, kernel_size=3, stride=2)
        lens = self.compute_mask_length(lens, kernel_size=5, stride=3)
        mask = tf.sequence_mask(lens)
        return mask


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """

    def __init__(
        self,
        idim: int,
        odim: int,
        dropout_rate: float,
        pos_enc_class: tf.keras.layers.Layer,
        bias_regularizer="l2",
        kernel_regularizer="l2",
        **kwargs,
    ):
        """Construct an Conv2dSubsampling8 object."""
        super(Conv2dSubsampling8, self).__init__(**kwargs)
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, idim, 1)),
            tf.keras.layers.Conv2D(
                odim,
                3,
                2,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            ),
            ActivationLayer('relu'),
            tf.keras.layers.Conv2D(
                odim,
                3,
                2,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            ),
            ActivationLayer('relu'),
            tf.keras.layers.Conv2D(
                odim,
                3,
                2,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
            ),
            ActivationLayer('relu'),
        ])

        # input shape: odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2
        self.out = tf.keras.layers.Dense(
            odim,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Subsample x.
        Args:
            inputs:
                x (tf.Tensor): Input tensor (#batch, time, idim).
                offset (tf.Tensor)
            x_mask (tf.Tensor): Input mask (#batch, time, 1 ).
        Returns:
            tf.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            tf.Tensor: Subsampled mask (#batch, time', 1),
                where time' = time // 8.
            tf.Tensor: positional encoding
        """

        x, offset = inputs
        x = tf.expand_dims(x, axis=3)  # (b, t, f, 1)
        x = self.conv(x)
        x_shape = tf.shape(x)
        b, t, f, c = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        x = self.out(tf.reshape(x, [b, t, f * c]))
        if training:
            x = tf.nn.dropout(x, self.dropout_rate)
        x, pos_emb = self.pos_enc((x, offset), training=training)
        # return x, pos_emb, mask[:, :-2:2, :][:, :-2:2, :][:, :-2:2, :]
        return x, pos_emb

    def get_mask(self, lens, mask=None):
        lens = self.compute_conv_length(lens, kernel_size=3, stride=2)
        lens = self.compute_conv_length(lens, kernel_size=3, stride=2)
        lens = self.compute_conv_length(lens, kernel_size=3, stride=2)
        return tf.sequence_mask(lens)
