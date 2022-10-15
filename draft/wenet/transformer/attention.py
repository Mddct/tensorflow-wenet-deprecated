"""Multi-Head Attention layer definition."""

import math
from typing import Optional, Tuple

import tensorflow as tf
from wenet.utils.common import mask_softmax


def GetFanInFanOut(shape):

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    receptive_field_size = 1
    if len(shape) > 2:
        receptive_field_size = tf.math.cumprod(shape[2:])[-1]
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def XavierUniform(shape, dtype, scale=1.0, method='xavier', seed=None):
    """Xavier initialization (x = sqrt(6. / (in + out)); scale*[-x, x])."""
    if not shape:
        raise ValueError(
            '\'shape\' must not be \'None\' or 0 for XavierUniform')
    fan_in, fan_out = GetFanInFanOut(shape)
    if method == 'xavier':
        limit = math.sqrt(6. / (fan_in + fan_out))
    else:
        assert method == 'geo_mean_xavier'
        limit = math.sqrt(3. / math.sqrt(fan_in * fan_out))
    return scale * tf.random.uniform(shape, -limit, limit, dtype, seed=seed)


class MultiHeadedAttention(tf.keras.layers.Layer):
    """Multi-Head Attention layer.
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.n_feat = n_feat

        self.linear_q = tf.keras.layers.Dense(n_feat)
        self.linear_k = tf.keras.layers.Dense(n_feat)
        self.linear_v = tf.keras.layers.Dense(n_feat)
        self.linear_out = tf.keras.layers.Dense(n_feat)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def forward_qkv(
            self, query: tf.Tensor, key: tf.Tensor,
            value: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Transform query, key and value.
        Args:
            query (tf.Tensor): Query tensor (#batch, time1, size).
            key (tf.Tensor): Key tensor (#batch, time2, size).
            value (tf.Tensor): Value tensor (#batch, time2, size).
        Returns:
            tf.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            tf.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            tf.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).
        """
        n_batch = tf.shape(query)[0]
        shapes = [n_batch, -1, self.h, self.d_k]
        q = tf.reshape(self.linear_q(query), shapes)
        k = tf.reshape(self.linear_k(key), shapes)
        v = tf.reshape(self.linear_v(value), shapes)
        q = tf.transpose(q, [0, 2, 1, 3])  # (batch, head, time1, d_k)
        k = tf.transpose(k, [0, 2, 1, 3])  # (batch, head, time2, d_k)
        v = tf.transpose(v, [0, 2, 1, 3])  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(self,
                          value: tf.Tensor,
                          scores: tf.Tensor,
                          mask: Optional[tf.Tensor] = None,
                          training: bool = True) -> tf.Tensor:
        """Compute attention context vector.
        Args:
            value (tf.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (tf.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (tf.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2)
        Returns:
            tf.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).
        """
        n_batch = tf.shape(value)[0]
        # mask = tf.equal(tf.expand_dims(mask, axis=1), 0)
        # mask = mask[:, :, :, :tf.shape(scores)[-1]]
        # scores = tf.where(mask, -float('inf'), scores)
        # attn = tf.nn.softmax(scores, axis=-1)  # (batch, head, time1, time2)

        if training:
            mask = tf.expand_dims(mask, axis=1)  # [B,1, *, time2]
            attn = mask_softmax(scores, mask, name='attention_weights')
        else:
            # NOTE: one uttrance don't need a mask
            # but for batch streamming inference mask is necessary
            # TODO: fix
            attn = tf.nn.softmax(scores, axis=-1, name='attention_weights')
        p_attn = self.dropout(attn, training)
        x = tf.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = tf.reshape(
            tf.transpose(x, [0, 1, 3, 2]),
            [n_batch, -1, self.h * self.d_k])  # (batch, time1, d_model)
        return self.linear_out(x)  # (batch, time1, d_model)

    def call(self,
             query: tf.Tensor,
             key: tf.Tensor,
             value: tf.Tensor,
             mask: Optional[tf.Tensor] = None,
             pos_emb: Optional[tf.Tensor] = None,
             cache: Optional[tf.Tensor] = None,
             training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute scaled dot product attention.
        Args:
            query (tf.Tensor): Query tensor (#batch, time1, size).
            key (tf.Tensor): Key tensor (#batch, time2, size).
            value (tf.Tensor): Value tensor (#batch, time2, size).
            mask (tf.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.
            cache (tf.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            tf.Tensor: Output tensor (#batch, time1, d_model).
            tf.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q, k, v = self.forward_qkv(query, key, value)
        if not training and cache is not None:
            key_cache, value_cache = tf.split(cache,
                                              cache.size(-1) // 2,
                                              axis=-1)
            k = tf.concat([key_cache, k], axis=2)
            v = tf.concat([value_cache, v], axis=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = tf.concat((k, v), axis=-1)
        depth = self.d_k**-0.5
        scores = tf.matmul(q * depth, tf.transpose(k, [0, 1, 3, 2]))
        return self.forward_attention(v, scores, mask), new_cache

    def get_config(self):
        conf = super(MultiHeadedAttention, self).get_config()
        conf.update({"head": self.h})
        conf.update({"feats": self.n_feat})
        conf.update({"linear_q": self.linear_q.get_config()})
        conf.update({"linear_k": self.linear_q.get_config()})
        conf.update({"linear_v": self.linear_q.get_config()})
        conf.update({"dropout": self.dropout.get_config()})

        return conf


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        # linear transformation for positional encoding
        self.linear_pos = tf.keras.layers.Dense(n_feat)
        self.pos_bias_u = tf.Variable(
            initial_value=XavierUniform(shape=[self.h, self.d_k],
                                        dtype=tf.float32),
            trainable=True,
        )
        self.pos_bias_v = tf.Variable(
            initial_value=XavierUniform(shape=[self.h, self.d_k],
                                        dtype=tf.float32),
            trainable=True,
        )

    def rel_shift_(self, x, zero_triu: bool = False):
        x_size = tf.shape(x)

        x = tf.pad(x, [[0, 0], [1, 0], [0, 0], [0, 0]])
        x = tf.reshape(x, [x_size[1] + 1, x_size[0], x_size[2], x_size[3]])
        x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, x_size)
        if zero_triu:
            ones = tf.ones([x_size[2], x.size[3]])
            x = tf.linalg.band_part(ones, -1, x.size(3) - x.size(2))

        return x

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        pos_emb: Optional[tf.Tensor] = None,
        cache: Optional[tf.Tensor] = None,
        training: bool = True,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (tf.Tensor): Query tensor (#batch, time1, size).
            key (tf.Tensor): Key tensor (#batch, time2, size).
            value (tf.Tensor): Value tensor (#batch, time2, size).
            mask (tf.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2)
            pos_emb (tf.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (tf.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            tf.Tensor: Output tensor (#batch, time1, d_model).
            tf.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = tf.transpose(q, [0, 2, 1, 3])  # (batch, time1, head, d_k)

        if not training and cache is not None:
            key_cache, value_cache = tf.split(cache,
                                              tf.shape(cache)[-1] // 2,
                                              axis=-1)
            k = tf.concat([key_cache, k], axis=2)
            v = tf.concat([value_cache, v], axis=2)

        new_cache = tf.concat((k, v), axis=-1)

        n_batch_pos = tf.shape(pos_emb)[0]
        p = tf.reshape(self.linear_pos(pos_emb),
                       [n_batch_pos, -1, self.h, self.d_k])
        p = tf.transpose(p, [0, 2, 1, 3])  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = tf.transpose(q + self.pos_bias_u, [0, 2, 1, 3])
        # (batch, head, time1, d_k)
        q_with_bias_v = tf.transpose(q + self.pos_bias_v, [0, 2, 1, 3])

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        depth = self.d_k**0.5
        matrix_ac = tf.matmul(q_with_bias_u * depth,
                              tf.transpose(k, [0, 1, 3, 2]))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = tf.matmul(q_with_bias_v * depth,
                              tf.transpose(p, [0, 1, 3, 2]))
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift_(matrix_bd)

        scores = (matrix_ac + matrix_bd)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask,
                                      training=training), new_cache
