#pragma once
#include <stdio.h>
#include <vector>
using namespace std;

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

struct Tensor {
  size_t n = 0;
  int ndim = 0;
  int shape[4];
  float *buf = nullptr;
  Tensor(const vector<int> &shape_);
  Tensor(float *data, const vector<int> &shape_);

  ~Tensor();
  size_t get_elem();
  void reshape(const vector<int> &shape_);
};
