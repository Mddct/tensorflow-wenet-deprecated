# g++ -std=c++17 -shared kaldi_fbank_kernels.cc  -o kaldi_fbank_kernels.cc.so  -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
import tensorflow as tf
from wenet.tfaudio  import fbank

output = fbank(tf.ones(640, dtype=tf.float32), 80, 25, 10, 0.0, 16000)

print(output)
