from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

gen_x_op = load_library.load_op_library(
    resource_loader.get_path_to_datafile("kaldi_fbank_kernels.cc.so"))
