import tensorflow as tf
from wenet.tfaudio.ops import gen_x_op


def fbank(waveform: tf.Tensor,
          num_mel_bins: int,
          frame_length: int,
          frame_shift: int,
          dither: float,
          sample_rate: int,
          energy_floor: float = 0.0,
          method="kaldi") -> tf.Tensor:
    """
    waveform: [samples]
    method:
        (1) kaldi
        (2) TODO: tf.signal which can run on gpu later 
    """
    assert method == "kaldi"

    _ = energy_floor
    return gen_x_op.kaldi_fbank_op(waveform, sample_rate, frame_length,
                                   frame_shift, num_mel_bins,
                                   dither)  # [frames, num_mel_bins]
