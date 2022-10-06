
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"

#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/kernels/conv_ops.h"

using namespace tensorflow;

template <typename Device, typename T>
class MyConv2DOp : public Conv2DOp<T> {
public:
  explicit MyConv2DOp(OpKernelConstruction *context) : Conv2DOp<T>(context) {
    // OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));

    // OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu",
    // &use_cudnn_)); cudnn_use_autotune_ = CudnnUseAutotune();
  }

  void Compute(OpKernelContext* context)  {
   }



  TF_DISALLOW_COPY_AND_ASSIGN(MyConv2DOp);
};
