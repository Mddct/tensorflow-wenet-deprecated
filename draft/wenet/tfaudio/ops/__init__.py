from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

#  g++ -std=c++17 -shared kaldi_fbank_kernels.cc  resample.cc  speed_kernels.cc -o tfaudio.so   -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
gen_x_op = load_library.load_op_library(
    resource_loader.get_path_to_datafile("tfaudio.so"))
