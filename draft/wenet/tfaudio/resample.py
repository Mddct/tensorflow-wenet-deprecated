import math
from typing import Optional

import tensorflow as tf


def _get_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    gcd: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interpolation",
    beta: Optional[float] = None,
    dtype=tf.float32,
):

    # if not (int(orig_freq) == orig_freq and int(new_freq) == new_freq):
    #     raise Exception(
    #         "Frequencies must be of integer type to ensure quality resampling computation. "
    #         "To work around this, manually convert both frequencies to integer values "
    #         "that maintain their resampling rate ratio before passing them into the function. "
    #         "Example: To downsample a 44100 hz waveform by a factor of 8, use "
    #         "`orig_freq=8` and `new_freq=1` instead of `orig_freq=44100` and `new_freq=5512.5`. "
    #         "For more information, please refer to https://github.com/pytorch/audio/issues/1487."
    #     )

    if resampling_method not in ["sinc_interpolation", "kaiser_window"]:
        raise ValueError(
            "Invalid resampling method: {}".format(resampling_method))

    orig_freq = orig_freq // gcd
    new_freq = new_freq // gcd
    orig_freq = tf.cast(orig_freq, dtype=tf.float32)
    new_freq = tf.cast(new_freq, dtype=tf.float32)
    lowpass_filter_width = tf.cast(lowpass_filter_width, dtype=tf.float32)

    # tf.debugging.Assert(
    #     # tf.constant(lowpass_filter_width) > 0,
    #     False,
    #     [''],
    # )
    # "Low pass filter width should be positive.")
    if lowpass_filter_width <= 0.0:
        raise ValueError("Low pass filter width should be positive.")
    base_freq = tf.math.minimum(orig_freq, new_freq)
    # This will perform antialiasing filtering by removing the highest frequencies.
    # At first I thought I only needed this when downsampling, but when upsampling
    # you will get edge artifacts without this, as the edge is equivalent to zero padding,
    # which will add high freq artifacts.
    # base_freq *= rolloff
    base_freq *= rolloff
    # The key idea of the algorithm is that x(t) can be exactly reconstructed from x[i] (tensor)
    # using the sinc interpolation formula:
    #   x(t) = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - t))
    # We can then sample the function x(t) with a different sample rate:
    #    y[j] = x(j / new_freq)
    # or,
    #    y[j] = sum_i x[i] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))

    # We see here that y[j] is the convolution of x[i] with a specific filter, for which
    # we take an FIR approximation, stopping when we see at least `lowpass_filter_width` zeros crossing.
    # But y[j+1] is going to have a different set of weights and so on, until y[j + new_freq].
    # Indeed:
    # y[j + new_freq] = sum_i x[i] sinc(pi * orig_freq * ((i / orig_freq - (j + new_freq) / new_freq))
    #                 = sum_i x[i] sinc(pi * orig_freq * ((i - orig_freq) / orig_freq - j / new_freq))
    #                 = sum_i x[i + orig_freq] sinc(pi * orig_freq * (i / orig_freq - j / new_freq))
    # so y[j+new_freq] uses the same filter as y[j], but on a shifted version of x by `orig_freq`.
    # This will explain the F.conv1d after, with a stride of orig_freq.
    width = tf.math.ceil(lowpass_filter_width * orig_freq / base_freq)
    # If orig_freq is still big after GCD reduction, most filters will be very unbalanced, i.e.,
    # they will have a lot of almost zero values to the left or to the right...
    # There is probably a way to evaluate those filters more efficiently, but this is kept for
    # future work.

    idx = tf.reshape(tf.range(-width, width + orig_freq, dtype=dtype),
                     [1, 1, -1]) / orig_freq

    t = tf.range(0, -new_freq, -1, dtype=dtype)[:, None, None] / new_freq + idx
    t *= base_freq
    t = tf.clip_by_value(t, -lowpass_filter_width, lowpass_filter_width)

    # we do not use built in torch windows here as we need to evaluate the window
    # at specific positions, not over a regular grid.
    if resampling_method == "sinc_interpolation":
        window = tf.math.cos(t * math.pi / lowpass_filter_width / 2)**2
    else:
        # kaiser_window
        if beta is None:
            beta = 14.769656459379492
        beta_tensor = tf.constant(beta, dtype=float)
        window = tf.math.bessel_i0(beta_tensor * tf.math.sqrt(1 - (
            t / lowpass_filter_width)**2)) / tf.math.bessel_i0(beta_tensor)

    t *= math.pi

    scale = base_freq / orig_freq
    if dtype is None:
        dtype = tf.float32
    kernels = tf.where(t == 0, 1.0, tf.math.sin(t) / t)
    kernels *= window * tf.cast(scale, dtype=tf.float32)

    width = tf.cast(width, dtype=tf.int32)
    return kernels, width


def _apply_sinc_resample_kernel(
    waveform: tf.Tensor,
    orig_freq: int,
    new_freq: int,
    gcd: int,
    kernel: tf.Tensor,
    width: int,
):
    if not (waveform.dtype == tf.float32 or waveform.dtype == tf.float64):
        raise TypeError(
            f"Expected floating point type for waveform tensor, but received {waveform.dtype}."
        )

    orig_freq = orig_freq // gcd
    new_freq = new_freq // gcd

    # pack batch
    shape = tf.shape(waveform)
    waveform = tf.reshape(waveform, [-1, shape[-1]])

    new_shape = tf.shape(waveform)
    num_wavs, length = new_shape[0], new_shape[1]
    waveform = tf.pad(waveform, [[0, 0], [width, width + orig_freq]])
    unfold = tf.image.extract_patches(images=tf.expand_dims(
        tf.expand_dims(waveform, 2), 3),
                                      sizes=[1, kernel.shape[-1], 1, 1],
                                      strides=[1, orig_freq, 1, 1],
                                      rates=[1, 1, 1, 1],
                                      padding='VALID')
    unfold = tf.expand_dims(unfold, 2)
    # conv1d
    resampled = tf.squeeze(tf.reduce_sum(unfold * kernel, axis=-1), -1)
    # resampled = tf.nn.conv1d(
    #     # tf.transpose(waveform[:, None], [0, 2, 1]),
    #     tf.transpose(tf.expand_dims(waveform, 1), [0, 2, 1]),
    #     # waveform[:, None],
    #     # waveform[:, None],
    #     tf.transpose(kernel, [2, 1, 0]),
    #     stride=orig_freq.numpy(),
    #     padding='VALID',
    #     # data_format="NWC")

    resampled = tf.reshape(resampled, [num_wavs, -1])
    target_length = int(math.ceil(new_freq * length / orig_freq))
    resampled = resampled[..., :target_length]

    # unpack batch
    n_shape = tf.concat([shape[:-1], tf.shape(resampled[-1:])], axis=0)
    resampled = tf.reshape(resampled, n_shape)
    # resampled = tf.reshape(resampled, [shape[:-1], tf.shape(resampled)[-1:]])
    return resampled


def eager_resample(
    waveform: tf.Tensor,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interpolation",
    beta: Optional[float] = None,
) -> tf.Tensor:
    r"""Resamples the waveform at the new frequency using bandlimited interpolation. :cite:`RESAMPLE`.
    .. devices:: CPU CUDA
    .. properties:: Autograd TorchScript
    Note:
        ``transforms.Resample`` precomputes and reuses the resampling kernel, so using it will result in
        more efficient computation if resampling multiple waveforms with the same resampling parameters.
    Args:
        waveform (Tensor): The input signal of dimension `(..., time)`
        orig_freq (int): The original frequency of the signal
        new_freq (int): The desired frequency
        lowpass_filter_width (int, optional): Controls the sharpness of the filter, more == sharper
            but less efficient. (Default: ``6``)
        rolloff (float, optional): The roll-off frequency of the filter, as a fraction of the Nyquist.
            Lower values reduce anti-aliasing, but also reduce some of the highest frequencies. (Default: ``0.99``)
        resampling_method (str, optional): The resampling method to use.
            Options: [``"sinc_interpolation"``, ``"kaiser_window"``] (Default: ``"sinc_interpolation"``)
        beta (float or None, optional): The shape parameter used for kaiser window.
    Returns:
        Tensor: The waveform at the new frequency of dimension `(..., time).`
    """

    # tf.debugging.Assert(
    #     orig_freq > 0, [orig_freq],
    #     "Original frequency and desired frequecy should be positive")
    if orig_freq == new_freq:
        return waveform

    gcd = tf.experimental.numpy.gcd(orig_freq, new_freq)
    # gcd = math.gcd(int(orig_freq), int(new_freq))

    kernel, width = _get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        beta,
        waveform.dtype,
    )
    resampled = _apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd,
                                            kernel, width)
    return resampled


# a = tf.ones([1, 200], dtype=tf.float32)


# print(
#     eager_resample(a, tf.constant(8000, dtype=tf.int32),
#                    tf.constant(16000, dtype=tf.int32)))
def resample_fn(waveform: tf.Tensor, ori_freq: tf.Tensor, new_freq: tf.Tensor):
    return tf.py_function(eager_resample, [waveform, ori_freq, new_freq],
                          waveform.dtype)
