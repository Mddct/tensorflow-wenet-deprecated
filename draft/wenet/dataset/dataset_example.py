import tensorflow as tf
from wenet.tfaudio import fbank, resample


def read_wav(path):
    return tf.io.read_file(path)


def decode(raw):
    wav, sr = tf.audio.decode_wav(raw)
    return tf.squeeze(wav), sr


bins = 80


def feature(audio):

    audio = audio * (1 << 15)
    return fbank(audio,
                 num_mel_bins=bins,
                 frame_length=25,
                 frame_shift=10,
                 dither=0.1,
                 sample_rate=16000)


wavs = ["1.wav", "test.wav", "test.wav", "test.wav"]
dataset = tf.data.Dataset.from_tensor_slices(wavs)
dataset = dataset.map(read_wav)
dataset = dataset.map(decode)
dataset = dataset.map(lambda waveform, sr: resample.resample_fn(
    waveform, sr, tf.constant(16000, dtype=tf.int32)))
# dataset.filter .... resampel ... rir..... ....speed....
dataset = dataset.map(feature)
dataset = dataset.padded_batch(batch_size=2,
                               padded_shapes=[None, None],
                               padding_values=0.0,
                               drop_remainder=True)

for batch in dataset:
    print(batch.shape)
