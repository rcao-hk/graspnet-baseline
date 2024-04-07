// Author: Rui Cao

#pragma once
#include <torch/extension.h>

at::Tensor rectangular_query(at::Tensor new_xyz, at::Tensor xyz, at::Tensor rot, at::Tensor size,
                      const int nsample);