#include "resample.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include <vector>
#include <algorithm>

REGISTER_OP("SpeedOp")
    .Input("input_data: float")
    .Input("sample_rate: int32")
    .Input("resample_freq: int32")
    .Attr("lowpass_filter_width: int = 1")
    .Output("output: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
      return Status::OK();
    });

class SpeedOp : public OpKernel {
 public:
  explicit SpeedOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("lowpass_filter_width", &lowpass_filter_width_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    OP_REQUIRES(context, input_tensor.dims() == 1,
                errors::InvalidArgument("input signal must be 1-dimensional",
                                        input_tensor.shape().DebugString()));
    const Tensor& sample_rate_tensor = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(sample_rate_tensor.shape()),
                errors::InvalidArgument(
                    "Input sample_rate should be a scalar tensor, got ",
                    sample_rate_tensor.shape().DebugString(), " instead."));
    const Tensor& resample_rate_tensor = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(resample_rate_tensor.shape()),
                errors::InvalidArgument(
                    "Resample sample_rate should be a scalar tensor, got ",
                    resample_rate_tensor.shape().DebugString(), " instead."));
    const int sample_rate = static_cast<int>(sample_rate_tensor.scalar<int32_t>()());
    const int resample_freq = static_cast<int>(resample_rate_tensor.scalar<int32_t>()());
    const float* input_flat = input_tensor.flat<float>().data();
    const int L = input_tensor.dim_size(0);

    lowpass_cutoff_ = std::min(resample_freq / 2, sample_rate / 2);
    LinearResample cls_resample_(sample_rate, resample_freq,
                                     lowpass_cutoff_,
                                     lowpass_filter_width_);
    std::vector<float> waveform(L);
    for (int i = 0; i < L; i++){
        waveform[i] = static_cast<float>(input_flat[i]);
    }
    std::vector<float> downsampled_wave;
    cls_resample_.Resample(waveform, false, &downsampled_wave);
    int output_length = downsampled_wave.size();
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({1, output_length}),
                                                     &output_tensor));
    float* output_flat = output_tensor->flat<float>().data();
    for (int j = 0; j < output_length; j++)
        output_flat[j] = downsampled_wave[j];

    std::vector<float>().swap(downsampled_wave);
    std::vector<float>().swap(waveform);
    cls_resample_.Reset();
   }

 private:
  float lowpass_cutoff_;
  int lowpass_filter_width_;
};

REGISTER_KERNEL_BUILDER(Name("SpeedOp").Device(DEVICE_CPU), SpeedOp);
