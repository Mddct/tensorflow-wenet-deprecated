import tensorflow as tf
from wenet.tfaudio import fbank, speed


def read_wav(path):
    # support read from s3
    raw = tf.io.read_file(path)
    wav, sr = tf.audio.decode_wav(raw)

    # assume one channel
    return tf.squeeze(wav), sr


# from yaml
bins = 80


def feature(audio):

    audio = audio * (1 << 15)
    audio = tf.squeeze(audio)
    return fbank(audio,
                 num_mel_bins=bins,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.1,
                 sample_rate=16000)


sample_rate = tf.constant(16000, dtype=tf.int32)
wavs = ["1.wav", "test.wav", "test.wav", "test.wav"] * 10000
dataset = tf.data.Dataset.from_tensor_slices(wavs)

# # dataset.filter .... resampel ... rir..... ....speed....
# == torchaudio.load
dataset = dataset.map(read_wav, num_parallel_calls=tf.data.AUTOTUNE)
# TODO: resample
# == torchaudio effects speed
dataset = dataset.map(
    lambda waveform, sr: speed(waveform, sr, tf.constant([0.9, 1., 1.1])),
    tf.data.AUTOTUNE)
# == torchadio fbank
dataset = dataset.map(feature)

# batch
dataset = dataset.padded_batch(batch_size=4,
                               padded_shapes=[None, None],
                               padding_values=0.0,
                               drop_remainder=True)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

for batch in dataset:
    print(batch.shape)
