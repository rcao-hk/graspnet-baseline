// Author: Rui Cao

#include "rectangular_query.h"
#include "utils.h"

void query_rectangular_point_kernel_wrapper(int b, int n, int m, const float *size,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, const float *rot, int *idx);

at::Tensor rectangular_query(at::Tensor new_xyz, at::Tensor xyz, at::Tensor rot, at::Tensor size,
                      const int nsample) {
  CHECK_CONTIGUOUS(new_xyz);
  CHECK_CONTIGUOUS(xyz);
  CHECK_CONTIGUOUS(rot);
  CHECK_CONTIGUOUS(size);
  CHECK_IS_FLOAT(new_xyz);
  CHECK_IS_FLOAT(xyz);
  CHECK_IS_FLOAT(rot);
  CHECK_IS_FLOAT(size);

  if (new_xyz.type().is_cuda()) {
    CHECK_CUDA(xyz);
    CHECK_CUDA(rot);
    CHECK_CUDA(size);
  }

  at::Tensor idx =
      torch::zeros({new_xyz.size(0), new_xyz.size(1), nsample},
                   at::device(new_xyz.device()).dtype(at::ScalarType::Int));

  if (new_xyz.type().is_cuda()) {
    query_rectangular_point_kernel_wrapper(xyz.size(0), xyz.size(1), new_xyz.size(1),
                                    size.data<float>(), nsample, new_xyz.data<float>(),
                                    xyz.data<float>(), rot.data<float>(), idx.data<int>());
  } else {
    TORCH_CHECK(false, "CPU not supported");
  }

  return idx;
}