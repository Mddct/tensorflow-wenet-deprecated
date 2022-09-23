# g++ -std=c++17 -shared kaldi_fbank_kernels.cc  -o kaldi_fbank_kernels.cc.so  -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2 -D_GLIBCXX_USE_CXX11_ABI=0
import tensorflow as tf
import time

kaldi = tf.load_op_library("kaldi_fbank_kernels.cc.so")

now = time.time()
output = kaldi.kaldi_fbank_op(tf.ones(640, dtype=tf.float32))

now = time.time()
output = kaldi.kaldi_fbank_op(2*tf.ones(640, dtype=tf.float32))
print(output)


print(time.time()-now)
