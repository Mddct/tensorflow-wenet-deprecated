import tensorflow as tf


def subsequent_mask(size: tf.Tensor) -> tf.Tensor:
    input = tf.ones((size, size), dtype=tf.bool)
    return tf.linalg.band_part(input, -1, 0)


def subsequent_chunk_mask(
        size: tf.Tensor,
        chunk_size: tf.Tensor,
        num_left_chunks: tf.Tensor = tf.constant(-1, dtype=tf.int32),
):

    index = tf.range(0, size)

    mask_seq = tf.minimum(((index // chunk_size) + 1) * chunk_size, size)
    mask_1 = tf.sequence_mask(mask_seq, maxlen=size)

    def limit_left_fn():
        left_mask = tf.maximum(
            (index // chunk_size - num_left_chunks) * chunk_size, 0)

        mask_2 = tf.sequence_mask(left_mask, maxlen=size)
        return mask_1 & (~mask_2)

    return tf.cond(
        tf.not_equal(num_left_chunks, -1),
        limit_left_fn,
        lambda: mask_1,
    )


@tf.function
def add_optional_chunk_mask(xs: tf.Tensor, masks: tf.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int, static_chunk_size: int,
                            num_decoding_left_chunks: int):

    if use_dynamic_chunk:
        max_len = tf.shape(xs)[1]
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = tf.constant(-1, dtype=tf.int64)
        elif decoding_chunk_size > 0:
            chunk_size = tf.convert_to_tensor(decoding_chunk_size,
                                              dtype=tf.int64)
            num_left_chunks = tf.convert_to_tensor(num_decoding_left_chunks,
                                                   dtype=tf.int64)
        else:
            chunk_size = tf.random.uniform(shape=[],
                                           minval=1,
                                           maxval=max_len,
                                           dtype=tf.int64)
            num_left_chunks = tf.constant(-1, dtype=tf.int64)

            if chunk_size > max_len // 2:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = tf.random.uniform(shape=[],
                                                        minval=0,
                                                        maxval=max_left_chunks,
                                                        dtype=tf.int64)

    elif static_chunk_size > 0:
        num_left_chunks = tf.convert_to_tensor(num_decoding_left_chunks,
                                               dtype=tf.int64)
        chunk_size = tf.convert_to_tensor(static_chunk_size)
    else:
        return masks

    chunk_masks = subsequent_chunk_mask(
        tf.shape(xs)[1], chunk_size, num_left_chunks)  # (L, L)
    masks = tf.transpose(masks, [0, 2, 1])  # [B, 1, T]
    chunk_masks = tf.expand_dims(chunk_masks, 0)  #[1,L, L]
    chunk_masks = masks & chunk_masks  # (B, L, L)

    return chunk_masks


def make_pad_mask(lengths: tf.Tensor, max_len=None):
    return ~tf.sequence_mask(lengths, maxlen=max_len)


def make_no_pad_mask(lengths: tf.Tensor, max_len):
    return tf.sequence_mask(lengths)


@tf.function
def get_next_cache_start(required_cache_size: tf.Tensor,
                         attention_key_size: tf.Tensor):
    if required_cache_size < 0:
        next_cache_start = tf.constant(0, dtype=required_cache_size.dtype)
    elif required_cache_size == 0:
        next_cache_start = attention_key_size
    else:
        next_cache_start = tf.maximum(attention_key_size - required_cache_size,
                                      0)
    return next_cache_start
