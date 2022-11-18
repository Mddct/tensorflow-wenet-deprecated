import tensorflow as tf
import tensorflow_io as tfio
from wenet.tfaudio import fbank, speed


def parse_line(line, symbol_table):
    # wav_path\tlabes
    # a.wav\t你 好 _we net
    # support read from s3
    lines = tf.strings.split(line, "\t")
    wav_path, ref = lines[0], lines[1]
    raw = tf.io.read_file(wav_path)

    # get wav
    wav, sr = tf.audio.decode_wav(raw)
    # get token ids
    tokens = symbol_table.lookup(tf.strings.split(ref, " "))

    # assume one channel
    return tf.squeeze(wav), sr, tokens


def compute_fbank(audio,
                  sr,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):

    audio = audio * (1 << 15)
    audio = tf.squeeze(audio)
    return fbank(audio,
                 num_mel_bins=num_mel_bins,
                 frame_length=frame_length,
                 frame_shift=frame_shift,
                 dither=dither,
                 sample_rate=16000)


def speed_perturb(waveform, sr, speeds):
    # speed = tf.constant([0.9, 1.0, 1.1], dtype=tf.float32)
    return speed(waveform, sr, speeds)


def spec_trim(feats, max_t=20):
    """ Trim tailing frames.
        ref: Rapid-U2++ [arxiv link]
        Args:
        Returns
    """
    max_frames = tf.shape(feats)[0]

    length = tf.random.uniform(shape=[],
                               minval=0,
                               maxval=max_t,
                               dtype=max_frames.dtype)

    return tf.cond(tf.math.less(length, max_frames // 2),
                   lambda: feats[:(max_frames - length), :], lambda: feats)


def spec_aug(feats, feats_length, augmenter=None):
    if augmenter is None:
        return feats
    return augmenter(feats, feats_length)


def filter(waveform,
           sr,
           labels,
           max_length=10240,
           min_length=10,
           token_max_length=200,
           token_min_length=1,
           min_output_input_ratio=0.0005,
           max_output_input_ratio=1.):
    num_frames = tf.shape(waveform)[0] / sr * 100
    toks_length = tf.shape(labels)[0]

    if num_frames < min_length:
        return False
    if num_frames > max_length:
        return False
    if toks_length < token_min_length:
        return False
    if toks_length > token_max_length:
        return False
    if num_frames != 0:
        frames_per_tok = tf.cast(toks_length, dtype=tf.float32) / tf.cast(
            num_frames, dtype=tf.float32)
        # frames_per_tok = tf.cast(toks_length, dtype=tf.float32) / tf.cast(
        # num_frames, dtype=tf.float32)
        if frames_per_tok < min_output_input_ratio:
            return False
        if frames_per_tok > max_output_input_ratio:
            return False
        return True
    else:
        return False


# sample_rate = tf.constant(16000, dtype=tf.int32)
# wavs = ["1.wav\t你 好", "test.wav\t你 们", "test.wav\t你", "test.wav\t你 好"]
# dataset = tf.data.Dataset.from_tensor_slices(wavs)

# # # dataset.filter .... resampel ... rir..... ....speed....
# # == torchaudio.load
# dataset = dataset.map(parse_line, num_parallel_calls=tf.data.AUTOTUNE)
# # TODO: resample
# # == torchaudio effects speed
# dataset = dataset.map(
#     lambda waveform, sr, labels:
#     (speed(waveform, sr, tf.constant([0.9, 1., 1.1])), sr, labels),
#     tf.data.AUTOTUNE)
# #
# # filter ...
# #

# # spec aug...
# # == torchadio fbank
# dataset = dataset.map(lambda waveform, sr, labels: (feature(waveform), labels))

# # batch
# dataset = dataset.padded_batch(batch_size=4,
#                                padded_shapes=([None, bins], [None]),
#                                padding_values=(0.0, tf.cast(0,
#                                                             dtype=tf.int64)),
#                                drop_remainder=True)
# dataset = dataset.prefetch(tf.data.AUTOTUNE)

# for feats, labels in dataset:
#     print(feats.shape, labels.shape, labels)
