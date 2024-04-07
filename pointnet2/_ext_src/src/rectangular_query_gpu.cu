// Author: Rui Cao

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"

__global__ void query_rectangular_point_kernel(int b, int n, int m, const float *__restrict__ size,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        const float *__restrict__ rot,
                                        int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  size += batch_index * n * 3;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  rot += batch_index * m * 9;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    float r0 = rot[j * 9 + 0];
    float r1 = rot[j * 9 + 1];
    float r2 = rot[j * 9 + 2];
    float r3 = rot[j * 9 + 3];
    float r4 = rot[j * 9 + 4];
    float r5 = rot[j * 9 + 5];
    float r6 = rot[j * 9 + 6];
    float r7 = rot[j * 9 + 7];
    float r8 = rot[j * 9 + 8];
    float length = size[j * 3 + 0];
    float width = size[j * 3 + 1];
    float height = size[j * 3 + 2];
    float half_length = length * 0.5f;
    float half_width = width * 0.5f;
    float half_height = height * 0.5f;

    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0] - new_x;
      float y = xyz[k * 3 + 1] - new_y;
      float z = xyz[k * 3 + 2] - new_z;
      float x_rot = r0 * x + r3 * y + r6 * z;
      float y_rot = r1 * x + r4 * y + r7 * z;
      float z_rot = r2 * x + r5 * y + r8 * z;
      if (fabs(x_rot) <= half_length && fabs(y_rot) <= half_width && fabs(z_rot) <= half_height) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

void query_rectangular_point_kernel_wrapper(int b, int n, int m, const float *size,
                                     int nsample, const float *new_xyz,
                                     const float *xyz, const float *rot, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_rectangular_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, size, nsample, new_xyz, xyz, rot, idx);

  CUDA_CHECK_ERRORS();
}

