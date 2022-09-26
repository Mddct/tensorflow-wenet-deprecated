import tensorflow as tf
from wenet.tfaudio import fbank, resample, speed


def read_wav(path):
    return tf.io.read_file(path)


def decode(raw):
    wav, sr = tf.audio.decode_wav(raw)
    return tf.transpose(wav, [1, 0]), sr


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
wavs = ["1.wav", "test.wav", "test.wav", "test.wav"] * 10
dataset = tf.data.Dataset.from_tensor_slices(wavs)
dataset = dataset.map(read_wav)
dataset = dataset.map(decode)
dataset = dataset.map(
    lambda waveform, sr: resample.resample_fn(waveform, sr, sample_rate))
dataset = dataset.map(lambda waveform: speed.speed_fn(
    waveform, sample_rate, tf.constant([0.9, 1., 1.1])))
# # dataset.filter .... resampel ... rir..... ....speed....
# dataset = dataset.map(feature)
dataset = dataset.padded_batch(batch_size=4,
                               padded_shapes=[None, None],
                               padding_values=0.0,
                               drop_remainder=True)

for batch in dataset:
    print(batch.shape)
