"""Subsampling layer definition."""

from typing import Tuple

import tensorflow as tf
from wenet.transformer.activations import ActivationLayer


class BaseSubsampling(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self,
                          offset: tf.Tensor,
                          size: tf.Tensor,
                          apply_dropout: bool = True,
                          training: bool = True) -> tf.Tensor:
        return self.pos_enc.position_encoding(offset, size, apply_dropout,
                                              training)


class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: tf.keras.layers.Layer):
        """Construct an linear object."""
        super().__init__()
        self.out = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[None, idim, 1]),
            tf.keras.layers.Dense(odim),
            tf.keras.layers.LayerNormalization(epsilon=1e-5),
            tf.keras.layers.Dropout(dropout_rate),
        ])

        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def call(
        self,
        x: tf.Tensor,
        x_mask: tf.Tensor,
        offset: tf.Tensor,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Input x.
        Args:
            x (tf.Tensor): Input tensor (#batch, time, idim).
            x_mask (tf.Tensor): Input mask (#batch, time, 1).
        Returns:
            tf.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            tf.Tensor: linear input mask (#batch, time', 1),
                where time' = time .
        """
        x = self.out(x)
        x = self.dropout(x, training=training)
        x, pos_emb = self.pos_enc(x, offset, training=training)
        return x, pos_emb, x_mask


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: tf.keras.layers.Layer):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Input(shape=[None, idim, 1]),
            tf.keras.layers.Conv2D(filters=odim, strides=2, kernel_size=3),
            ActivationLayer('relu'),
            tf.keras.layers.Conv2D(filters=odim, strides=2, kernel_size=3),
            ActivationLayer('relu'),
        ])

        # torch.nn.Conv2d(1, odim, 3, 2),
        # torch.nn.Conv2d(odim, odim, 3, 2),
        # input dim should  == odim * (((idim - 1) // 2 - 1) // 2)
        self.out = tf.keras.layers.Dense(odim)
        self.pos_enc = pos_enc_class
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) * 1 + (3 - 1) * 2
        self.right_context = 6

    def call(
        self,
        x: tf.Tensor,
        x_mask: tf.Tensor,
        offset: tf.Tensor,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Subsample x.
        Args:
            x (tf.Tensor): Input tensor (#batch, time, idim).
            x_mask (tf.Tensor): Input mask (#batch, time, 1).
        Returns:
            tf.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            tf.Tensor: Subsampled mask (#batch, time', 1),
                where time' = time // 4.
            tf.Tensor: positional encoding
        """
        x = tf.expand_dims(x, axis=3)  # (b, t, f, 1)
        # x = self.conv1(x)  # (b, t', f', 1)
        # x = tf.nn.relu(x)
        # x = self.conv2(x)  # (b, t'', f'', odim)
        x = self.conv(x)
        x_shape = tf.shape(x)
        b, t, f, c = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        x = self.out(tf.reshape(x, [b, t, f * c]))
        x, pos_emb = self.pos_enc(x, offset, training=training)
        return x, pos_emb, x_mask[:, :-2:2, :][:, :-2:2, :]


class Conv2dSubsampling6(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/6 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (tf.keras.layers.Layer): Custom position encoding layer.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: tf.keras.layers.Layer):
        """Construct an Conv2dSubsampling6 object."""
        super().__init__()
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, idim, 1)),
            tf.keras.layers.Conv2D(odim, 3, 2),
            ActivationLayer('relu'),
            tf.keras.layers.Conv2D(odim, 5, 3),
            ActivationLayer('relu')
        ])
        # input shape:  odim * ((idim-1)//2-2//3)
        self.out = tf.keras.layers.Dense(odim)
        self.pos_enc = pos_enc_class
        # 10 = (3 - 1) * 1 + (5 - 1) * 2
        self.subsampling_rate = 6
        self.right_context = 10

    def call(
        self,
        x: tf.Tensor,
        x_mask: tf.Tensor,
        offset: tf.Tensor,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Subsample x.
        Args:
            x (tf.Tensor): Input tensor (#batch, time, idim).
            x_mask (tf.Tensor): Input mask (#batch, time, 1).
        Returns:
            tf.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            tf.Tensor: Subsampled mask (#batch, time', 1),
                where time' = time // 6.
            tf.Tensor: positional encoding
        """
        x = tf.expand_dims(x, axis=3)  # (b, t, f, 1)
        # x = self.conv1(x)  # (b, t', f', 1)
        # x = tf.nn.relu(x)
        # x = self.conv2(x)  # (b, t'', f'', odim)
        x = self.conv(x)
        x_shape = tf.shape(x)
        b, t, f, c = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        x = self.out(tf.reshape(x, [b, t, f * c]))
        x, pos_emb = self.pos_enc(x, offset, training=training)
        return x, pos_emb, x_mask[:, :-2:2, :][:, :-4:3, :]


class Conv2dSubsampling8(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/8 length).
    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: tf.keras.layers.Layer):
        """Construct an Conv2dSubsampling8 object."""
        super().__init__()
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, idim, 1)),
            tf.keras.layers.Conv2D(odim, 3, 2),
            ActivationLayer('relu'),
            tf.keras.layers.Conv2D(odim, 3, 2),
            ActivationLayer('relu'),
            tf.keras.layers.Conv2D(odim, 3, 2),
            ActivationLayer('relu'),
        ])

        # input shape: odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2
        self.out = tf.keras.layers.Dense(odim)
        self.pos_enc = pos_enc_class
        self.subsampling_rate = 8
        # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
        self.right_context = 14

    def call(
        self,
        x: tf.Tensor,
        x_mask: tf.Tensor,
        offset: tf.Tensor,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Subsample x.
        Args:
            x (tf.Tensor): Input tensor (#batch, time, idim).
            x_mask (tf.Tensor): Input mask (#batch, time, 1 ).
        Returns:
            tf.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            tf.Tensor: Subsampled mask (#batch, time', 1),
                where time' = time // 8.
            tf.Tensor: positional encoding
        """

        x = tf.expand_dims(x, axis=3)  # (b, t, f, 1)
        x = self.conv(x)
        x_shape = tf.shape(x)
        b, t, f, c = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        x = self.out(tf.reshape(x, [b, t, f * c]))
        x, pos_emb = self.pos_enc(x, offset, training=training)
        return x, pos_emb, x_mask[:, :-2:2, :][:, :-2:2, :][:, :-2:2, :]
