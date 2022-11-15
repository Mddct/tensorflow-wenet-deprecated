"""Positonal Encoding Module."""

import math
from typing import Tuple

import numpy as np
import tensorflow as tf

# def positional_encoding(length, depth):
#     depth = depth // 2

#     positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
#     depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
#     angle_rads = positions * np.exp(depths * (-math.log(10000.0)))

#     pos_encoding = np.concatenate(
#         [np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

#     pos_encoding = np.reshape(
#         np.transpose(np.reshape(pos_encoding, [length, 2, depth]), [0, 2, 1]),
#         [length, -1])  # [B, T]
#     return tf.cast(pos_encoding, dtype=tf.float32)


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

        self.pe = tf.expand_dims(positional_encoding(max_len, d_model), axis=0)

    def call(self,
             inputs,
             training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        """Add positional encoding.
        Args:
            inputs:
                x (tf.Tensor): Input. Its shape is (batch, time, ...)
                offset (tf.tensor): position offset
        Returns:
            tf.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            tf.Tensor: for compatibility to RelPositionalEncoding
        """
        x, offset = inputs
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
            offset (tf.Tensor): start offset [B]
            size (tf.Tensor): required size of position encoding
        Returns:
            tf.Tensor: Corresponding encoding
        """
        # tf.assert_less(tf.reduce_max(offset) + size, self.max_len)
        index = tf.range(0, size) + tf.expand_dims(offset, axis=1)
        index = tf.where(index > 0, index, 0)
        pos_emb = tf.nn.embedding_lookup(self.pe[0], index)  # B X T X d_model

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
             inputs,
             training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute positional encoding.
        Args:
            inputs:
                  x (tf.Tensor): Input tensor (batch, time, `*`).
                  offset (tf.Tensor): [B]
        Returns:
            tf.Tensor: Encoded tensor (batch, time, `*`).
            tf.Tensor: Positional embedding tensor (1, time, `*`).
        """
        x, offset = inputs
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
             inputs,
             training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        """ Just return zero vector for interface compatibility
        """
        x, _ = inputs
        pos_emb = tf.zeros([1, tf.shape(x)[1], self.d_model], dtype=tf.float32)
        return self.dropout(x, training=training), pos_emb

    def position_encoding(
        self,
        offset: tf.Tensor,
        size: tf.Tensor,
    ) -> tf.Tensor:
        _ = offset
        return tf.zeros([1, size, self.d_model], dtype=tf.float32)


class EmbeddingSharedWeights(tf.keras.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, hidden_size):
        """Specify characteristic parameters of embedding layer.

        Args:
          vocab_size: Number of tokens in the embedding. (Typically ~32,000)
          hidden_size: Dimensionality of the embedding. (Typically 512 or 1024)
        """
        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def build(self, input_shape):
        """Build embedding layer."""
        with tf.name_scope("embedding_and_softmax"):
            # Create and initialize weights. The random normal initializer was chosen
            # arbitrarily, and works well.
            self.shared_weights = self.add_weight(
                "weights",
                shape=[self.vocab_size, self.hidden_size],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(
                    mean=0., stddev=self.hidden_size**-0.5))
        super(EmbeddingSharedWeights, self).build(input_shape)

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
        }

    def call(self, inputs, mode="embedding"):
        """Get token embeddings of inputs.

        Args:
          inputs: An int64 tensor with shape [batch_size, length]
          mode: string, a valid value is one of "embedding" and "linear".

        Returns:
          outputs: (1) If mode == "embedding", output embedding tensor, float32 with
          shape [batch_size, length, embedding_size]; (2) mode == "linear", output
          linear tensor, float32 with shape [batch_size, length, vocab_size].

        Raises:
          ValueError: if mode is not valid.
        """
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def _embedding(self, inputs):
        """Applies embedding based on inputs tensor."""
        with tf.name_scope("embedding"):
            # Create binary mask of size [batch_size, length]
            embeddings = tf.gather(self.shared_weights, inputs)
            # mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
            # embeddings *= tf.expand_dims(mask, -1)
            # Scale embedding by the sqrt of the hidden size
            embeddings *= self.hidden_size**0.5

            return embeddings

    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.

        Args:
          inputs: A float32 tensor with shape [batch_size, length, hidden_size]

        Returns:
          float32 tensor with shape [batch_size, length, vocab_size].
        """
        with tf.name_scope("presoftmax_linear"):
            batch_size = tf.shape(inputs)[0]
            length = tf.shape(inputs)[1]

            x = tf.reshape(inputs, [-1, self.hidden_size])
            logits = tf.matmul(x, self.shared_weights, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])
