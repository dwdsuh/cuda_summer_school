#include <cstring>
#include <stdlib.h>

#include "tensor.h"
#include "util.h"
using namespace std;

Tensor::Tensor(const vector<int> &shape_) {
  reshape(shape_);
  CHECK_CUDA(cudaMalloc(&buf, n * sizeof(float)));
}

Tensor::Tensor(float *data, const vector<int> &shape_) {
  reshape(shape_);
  CHECK_CUDA(cudaMalloc(&buf, n * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(buf, data, get_elem() * sizeof(float), cudaMemcpyHostToDevice));
}

Tensor::~Tensor() { CHECK_CUDA(cudaFree(buf)); }

size_t Tensor::get_elem() { return n; }

void Tensor::reshape(const vector<int> &shape_) {
  n = 1;
  ndim = shape_.size(); // ndim<=4
  for (int i = 0; i < ndim; i++) {
    shape[i] = shape_[i];
    n *= shape[i];
  }
}
