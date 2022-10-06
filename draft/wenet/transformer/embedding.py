"""Positonal Encoding Module."""

import math
from typing import Tuple

import numpy as np
import tensorflow as tf


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000**depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding.
    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length
    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 name="positional_encoding",
                 reverse: bool = False,
                 **kwargs):
        """Construct an PositionalEncoding object."""
        super().__init__(trainable=False, name=name, **kwargs)
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.max_len = max_len

        # self.pe = tf.zeros([self.max_len, self.d_model], dtype=tf.float32)
        # position = tf.expand_dims(tf.range(0, self.max_len, dtype=tf.float32),
        #                           axis=1)
        # div_term = tf.math.exp(
        #     tf.range(0, self.d_model, 2, dtype=tf.float32) *
        #     -(math.log(10000.0) / self.d_model))
        # self.pe[:, 0::2] = tf.math.sin(position * div_term)
        # self.pe[:, 1::2] = tf.math.cos(position * div_term)
        # self.pe = tf.squeeze(self.pe, axis=0)
        self.pe = tf.expand_dims(positional_encoding(max_len, d_model), axis=0)

    def call(self,
             x: tf.Tensor,
             offset: tf.Tensor,
             training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        """Add positional encoding.
        Args:
            x (tf.Tensor): Input. Its shape is (batch, time, ...)
            offset (tf.tensor): position offset
        Returns:
            tf.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            tf.Tensor: for compatibility to RelPositionalEncoding
        """

        pos_emb = self.position_encoding(offset,
                                         tf.shape(x)[1],
                                         False,
                                         training=training)
        x = x * self.xscale + pos_emb
        return self.dropout(x,
                            training=training), self.dropout(pos_emb,
                                                             training=training)

    def position_encoding(self,
                          offset: tf.Tensor,
                          size: tf.Tensor,
                          apply_dropout: bool = True,
                          training: bool = True) -> tf.Tensor:
        """ For getting encoding in a streaming fashion
        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.
        Args:
            offset (tf.Tensor): start offset
            size (tf.Tensor): required size of position encoding
        Returns:
            tf.Tensor: Corresponding encoding
        """
        tf.assert_less(tf.reduce_max(offset) + size, self.max_len)
        index = tf.add(tf.expand_dims(offset, axis=1), tf.range(0, size))
        index = tf.where(index > 0, index, 0)
        pos_emb = tf.nn.embedding_lookup(self.pe[0], index)
        # pos_emb = F.embedding(index, self.pe[0])  # B X T X d_model

        if apply_dropout:
            pos_emb = self.dropout(pos_emb, training)
        return pos_emb


class RelPositionalEncoding(PositionalEncoding):
    """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 max_len: int = 5000,
                 name='rel_positional_encoding'):
        """Initialize class."""
        super().__init__(d_model,
                         dropout_rate,
                         max_len,
                         reverse=True,
                         name=name)

    def call(self,
             x: tf.Tensor,
             offset: tf.Tensor,
             training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute positional encoding.
        Args:
            x (tf.Tensor): Input tensor (batch, time, `*`).
        Returns:
            tf.Tensor: Encoded tensor (batch, time, `*`).
            tf..Tensor: Positional embedding tensor (1, time, `*`).
        """
        x = x * self.xscale
        pos_emb = self.position_encoding(offset,
                                         tf.shape(x)[1],
                                         False,
                                         training=training)
        return self.dropout(x,
                            training=training), self.dropout(pos_emb,
                                                             training=training)


class NoPositionalEncoding(tf.keras.layers.Layer):
    """ No position encoding
    """

    def __init__(self,
                 d_model: int,
                 dropout_rate: float,
                 name='no_position_encoding'):
        super().__init__(name=name, trainable=False)
        self.d_model = d_model
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self,
             x: tf.Tensor,
             offset: tf.Tensor,
             training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        """ Just return zero vector for interface compatibility
        """
        pos_emb = tf.zeros([1, tf.shape(x)[1]], self.d_model)
        return self.dropout(x, training=training), pos_emb

    def position_encoding(
        self,
        offset: tf.Tensor,
        size: tf.Tensor,
    ) -> tf.Tensor:
        return tf.zeros([1, size], self.d_model)


# rel_pos = RelPositionalEncoding(
#     256,
#     0.1,
# )
# print(rel_pos.pe)
