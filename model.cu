#include <cmath>
#include <cstring>

#include "model.h"
#include "util.h"

extern int N;

// class BrainTumorModel(nn.Module):
//
//  def __init__(self):
//      super().__init__()
//      self.conv0 = nn.Sequential(
//          nn.Conv2d(1,128,kernel_size=3),
//          nn.InstanceNorm2d(128),
//          nn.MaxPool2d(2,2),
//          nn.ReLU()
//      )
//
//      self.conv1 = nn.Sequential(
//          nn.Conv2d(128,256,kernel_size=3),
//          nn.InstanceNorm2d(256),
//          nn.MaxPool2d(2,2),
//          nn.ReLU()
//      )
//
//      self.linear1 = nn.Linear(62,128)
//      self.linear2 = nn.Linear(128,64)
//      self.flat = nn.Flatten(1)
//      self.linear3 = nn.Linear(1015808,2)
//
//  def forward(self,x):
//      x = self.conv0(x)
//      x = self.conv1(x)
//      x = F.relu(self.linear1(x))
//      x = self.linear2(x)
//      x = self.flat(x)
//      x = self.linear3(x)
//
//      return x

#define MAX_BATCH_SIZE (256)
static Tensor *conv0_weight, *conv0_bias, *conv1_weight, *conv1_bias,
    *linear1_weight, *linear1_bias, *linear2_weight, *linear2_bias,
    *linear3_weight, *linear3_bias, *instanceNorm2d0_weight,
    *instanceNorm2d0_bias, *instanceNorm2d1_weight, *instanceNorm2d1_bias;

static Tensor *input, *output, *c1, *i1, *m1, *c2, *i2, *m2, *l1, *l2;
void initialize_model(const char *parameter_fname) {
  size_t m; // 2345922
  float *buf = (float *)read_binary(parameter_fname, &m);
  conv0_weight = new Tensor(buf, {128, 1, 3, 3});
  buf += 1152;
  conv0_bias = new Tensor(buf, {128});
  buf += 128;
  instanceNorm2d0_weight = new Tensor(buf, {128});
  buf += 128;
  instanceNorm2d0_bias = new Tensor(buf, {128});
  buf += 128;
  conv1_weight = new Tensor(buf, {256, 128, 3, 3});
  buf += 294912;
  conv1_bias = new Tensor(buf, {256});
  buf += 256;
  instanceNorm2d1_weight = new Tensor(buf, {256});
  buf += 256;
  instanceNorm2d1_bias = new Tensor(buf, {256});
  buf += 256;
  linear1_weight = new Tensor(buf, {62, 128});
  buf += 7936;
  linear1_bias = new Tensor(buf, {128});
  buf += 128;
  linear2_weight = new Tensor(buf, {128, 64});
  buf += 8192;
  linear2_bias = new Tensor(buf, {64});
  buf += 64;
  linear3_weight = new Tensor(buf, {1015808, 2});
  buf += 2031616;
  linear3_bias = new Tensor(buf, {2});
  buf += 2;

  int batch = std::min(N, MAX_BATCH_SIZE);
  input = new Tensor({batch, 1, 256, 256});
  output = new Tensor({batch, 2});
  c1 = new Tensor({batch, 128, 254, 254});
  i1 = new Tensor({batch, 128, 254, 254});
  m1 = new Tensor({batch, 128, 127, 127});
  c2 = new Tensor({batch, 256, 125, 125});
  i2 = new Tensor({batch, 256, 125, 125});
  m2 = new Tensor({batch, 256, 62, 62});
  l1 = new Tensor({batch, 256, 62, 128});
  l2 = new Tensor({batch, 256, 62, 64});
}
// Conv2D
// https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
// Size of in  = N * C_IN * H_IN * W_IN
// Size of out = N * C_OUT * (H_IN-K+1) * (W_IN-K+1)
// Weight : C_OUT * C_IN * K * K
// Bias : C_OUT

static void conv2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t);

// MaxPool2d
// https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
// size of in  = N * H_IN * W_IN
// size of out = N * (H / kH) * (W / kW)
static void maxpool2d(Tensor *in_t, Tensor *out_t, int kH, int kW);

// InstanceNorm2D
// https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm2d.html
// size of in  = N * C * H * W
// size of out = N * C * H * W
// weight : C
// bias : C
static void instancenorm2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                           Tensor *bias_t);

// Linear
// https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
// size of in  = N * H_IN
// size of out = N * H_OUT
// weight : H_OUT * H_IN
// bias : H_OUT
static void linear(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t);

// ReLU (inplace)
// https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
// size of in & out = N
static void relu(Tensor *inout_t);

void model_forward(float *inputN, float *outputN) {
  int batch = std::min(N, MAX_BATCH_SIZE);
  int batch_count = (N + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE; 
  double conv2d_t = 0.0;
  double instance_t = 0.0;
  double maxpool2d_t = 0.0;
  double relu_t  = 0.0;
  double linear_t = 0.0;                 
  double st;
  double memcp_t = 0.0;
  for (int idx = 0 ; idx < batch_count ; idx++) {
    fprintf(stderr, "batch [%d]...\n", idx);
    if ((idx + 1) == batch_count) {
      if (N % MAX_BATCH_SIZE != 0) batch = N % MAX_BATCH_SIZE;
      if (batch < input->shape[0]) {
        input->shape[0] = batch;
        output->shape[0] = batch;
        c1->shape[0] = batch;
        i1->shape[0] = batch;
        m1->shape[0] = batch;
        c2->shape[0] = batch;
        i2->shape[0] = batch;
        m2->shape[0] = batch;
        l1->shape[0] = batch;
        l2->shape[0] = batch;
      }
    }
    CHECK_CUDA(cudaMemcpy(input->buf, inputN + 256 * 256 * idx * batch,
                          256 * 256 * sizeof(float) * batch,
                          cudaMemcpyHostToDevice));
    conv2d(input, c1, conv0_weight, conv0_bias);
    instancenorm2d(c1, i1, instanceNorm2d0_weight, instanceNorm2d0_bias);
    maxpool2d(i1, m1, 2, 2);
    relu(m1);
    conv2d(m1, c2, conv1_weight, conv1_bias);
    instancenorm2d(c2, i2, instanceNorm2d1_weight, instanceNorm2d1_bias);
    maxpool2d(i2, m2, 2, 2);
    relu(m2);
    linear(m2, l1, linear1_weight, linear1_bias);
    relu(l1);
    linear(l1, l2, linear2_weight, linear2_bias);
    l2->reshape({batch, 1, 1015808});
    linear(l2, output, linear3_weight, linear3_bias);
    CHECK_CUDA(cudaMemcpy(outputN, output->buf, 2 * sizeof(float) * batch,
                          cudaMemcpyDeviceToHost));
    outputN += 2 * batch;
  }
  fprintf(stderr,"conv2d: %lf\n", conv2d_t);
  fprintf(stderr,"maxpool2d: %lf\n", maxpool2d_t);
  fprintf(stderr,"linear: %lf\n", linear_t);
  fprintf(stderr,"instance: %lf\n", instance_t);
  fprintf(stderr,"relu: %lf\n", relu_t);
  fprintf(stderr,"cudaMemcpy: %lf\n", memcp_t); 
}

__global__ void conv2d_kernel(float *in, float *out, float *weight, float *bias,
	               int K, int C_IN, int C_OUT, int H_IN, int W_IN,
		       int H_OUT, int W_OUT, int B){
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  
  int b = tidx / (C_OUT * H_OUT * W_OUT);
  int c_out = (tidx / (H_OUT * W_OUT)) % C_OUT;
  int h_out = (tidx / W_OUT) % H_OUT;
  int w_out = tidx % W_OUT;

  if (b >= B || c_out >= C_OUT || h_out >= H_OUT || w_out >= W_OUT) return;

  out += b * C_OUT * H_OUT * W_OUT;
  in += b * C_IN * H_IN * W_IN;
  out[c_out * H_OUT * W_OUT + h_out * W_OUT + w_out] = bias[c_out];
  for (int c_in = 0; c_in < C_IN; c_in++) {
    for (int kh = 0; kh < K; kh++) {
      for (int kw = 0; kw < K; kw++) {
        out[c_out * H_OUT * W_OUT + h_out * W_OUT + w_out] +=
            in[c_in * H_IN * W_IN + (h_out + kh) * W_IN + (w_out + kw)] *
            weight[c_out * C_IN * K * K + c_in * K * K + kh * K + kw];
      }
    }
  }
}

static void conv2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t) {
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  int K = weight_t->shape[2]; //=weight_t->shape[3];

  int C_IN = weight_t->shape[1];  //=in_t->shape[0];
  int C_OUT = weight_t->shape[0]; //=out_t->shape[0];

  int batch = in_t->shape[0];
  int H_IN = in_t->shape[2];
  int W_IN = in_t->shape[3];
  int H_OUT = H_IN - K + 1; //=out_t->shape[1];
  int W_OUT = W_IN - K + 1; //=out_t->shape[2];
  int n_thread = batch * C_OUT * H_OUT * W_OUT;
  dim3 blockDim(512);
  dim3 gridDim((n_thread + 511) / 512);

  conv2d_kernel<<<gridDim, blockDim>>>(in, out, weight, bias, K, C_IN, C_OUT, H_IN, W_IN, H_OUT, W_OUT, batch);

}


__global__ void instancenorm2d_kernel(float *in, float *out, float *weight, float *bias,
	                              int C, int H, int W, int B){
  //int c = blockDim.x * blockIdx.x + threadIdx.x;
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  int b = tidx / C;
  int c = tidx % C;
  
  if (b >= B || c >= C) return;
  float e = 0, v = 0;

  // Caculate mean
  for (int h = 0; h < H; h++) {
    for (int w = 0; w < W; w++) {
      e += in[b * C * H * W + c * H * W + h * W + w];
    }
  }
  //calc_mean_kernel<<<1,1>>>(in, H, W, c, &e);
  e /= H * W;

  // Caculate Variance
  for (int h = 0; h < H; h++) {
    for (int w = 0; w < W; w++) {
      v += (in[b * C * H * W + c * H * W + h * W + w] - e) * (in[b * C * H * W + c * H * W + h * W + w] - e);
    }
  }
  v /= H * W;

  for (int h = 0; h < H; h++) {
    for (int w = 0; w < W; w++) {
      out[b * C * H * W + c * H * W + h * W + w] =
          (in[b * C * H * W + c * H * W + h * W + w] - e) / sqrt(v + 1e-5) * weight[c] +
          bias[c];
    }
  }
}



static void instancenorm2d(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                           Tensor *bias_t) {
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  int B = in_t->shape[0];
  int C = in_t->shape[1]; //=out_t->shape[0];
  int H = in_t->shape[2]; //=out_t->shape[1];
  int W = in_t->shape[3]; //=out_t->shape[2];
  int n_thread = C * B;
  dim3 blockDim(512);
  dim3 gridDim((n_thread + 512 -1) / 512);

  instancenorm2d_kernel<<<gridDim, blockDim>>>(in, out, weight, bias, C, H, W, B);
}

__global__ void linear_kernel(float *in, float *out, float *weight, float *bias,
	                      int H_IN, int H_OUT, int N, int batch){
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  int b = tidx / (N * H_OUT);
  int n = (tidx / H_OUT) % N;
  int h_out = tidx % H_OUT;

  if (b >= batch || n >= N || h_out >= H_OUT) return;

  out[b * N * H_OUT + n * H_OUT + h_out] = bias[h_out];
  for (int h_in = 0; h_in < H_IN; h_in++) {
    out[b * N * H_OUT + n * H_OUT + h_out] +=
        in[b * N * H_IN + n * H_IN + h_in] * weight[h_out * H_IN + h_in];
  }
  /*
  for (int n = 0; n < N; n++) {
    for (int h_out = 0; h_out < H_OUT; h_out++) {
      out[n * H_OUT + h_out] = bias[h_out];
      for (int h_in = 0; h_in < H_IN; h_in++) {
        out[n * H_OUT + h_out] +=
            in[n * H_IN + h_in] * weight[h_out * H_IN + h_in];
      }
    }
  } */
}

static void linear(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                   Tensor *bias_t) {
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;

  int batch = in_t->shape[0];
  int H_IN = weight_t->shape[0];  // in_t의 마지막 차원
  int H_OUT = weight_t->shape[1]; // out_t의 마지막 차원

  int N = in_t->get_elem() / H_IN / batch ; //=out_t->get_elem()/H_OUT
  // get_elem() already include batch
  int n_thread = N * H_OUT * batch;
  dim3 blockDim(512);
  dim3 gridDim((n_thread + 512 - 1) / 512);
  linear_kernel<<<gridDim, blockDim>>>(in, out, weight, bias, H_IN, H_OUT, N, batch);
}

__global__ void maxpool2d_kernel(float *in, float *out, 
	int H_IN, int W_IN, int H_OUT, int W_OUT, int N, int kH, int kW, int batch){
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  int b = tidx / (N * H_OUT * W_OUT);
  int n = (tidx / (H_OUT * W_OUT)) % N;
  int h_out = (tidx / W_OUT) % H_OUT;
  int w_out = tidx % W_OUT;

  if (b >= batch || n >= N || h_out >= H_OUT || w_out >= W_OUT) return;

  out[b * N * H_OUT * W_OUT + n * H_OUT * W_OUT + h_out * W_OUT + w_out] =
      in[b * N * H_IN * W_IN + n * H_IN * W_IN + (h_out * kH) * H_IN + (w_out * kW)];
  for (int kh = 0; kh < kH; kh++)
    for (int kw = 0; kw < kW; kw++)
      out[b * N * H_OUT * W_OUT + n * H_OUT * W_OUT + h_out * W_OUT + w_out] =
          fmaxf(out[b * N * H_OUT * W_OUT + n * H_OUT * W_OUT + h_out * W_OUT + w_out],
                in[b * N * H_IN * W_IN + n * H_IN * W_IN + (h_out * kH + kh) * H_IN +
                   (w_out * kW + kw)]);
}

static void maxpool2d(Tensor *in_t, Tensor *out_t, int kH, int kW) {
  float *in = in_t->buf;
  float *out = out_t->buf;

  int H_IN = in_t->shape[2];
  int W_IN = in_t->shape[3];
  int H_OUT = H_IN / kH; // =out_t->shape[1];
  int W_OUT = W_IN / kW; // =out_t->shape[2];

  int batch = in_t->shape[0];
  int N = in_t->shape[1];
  int n_thread = batch * N * H_OUT * W_OUT;
  dim3 blockDim(512);
  dim3 gridDim((n_thread + 512 -1) / 512);
  maxpool2d_kernel<<<gridDim, blockDim>>>(in, out, H_IN, W_IN, H_OUT, W_OUT, N, kH, kW, batch);
}

__global__ void relu_kernel(float *inout, int N){
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n >= N) return;
  inout[n] = fmaxf(inout[n], 0);
}

static void relu(Tensor *inout_t) {
  float *inout = inout_t->buf;
  int N = inout_t->get_elem();
  dim3 blockDim(512);
  dim3 gridDim((N+511) / 512);
  relu_kernel<<<gridDim, blockDim>>>(inout, N);
}

void finalize_model() {
  delete (conv0_weight);
  delete (conv0_bias);
  delete (conv1_weight);
  delete (conv1_bias);
  delete (linear1_weight);
  delete (linear1_bias);
  delete (linear2_weight);
  delete (linear2_bias);
  delete (linear3_weight);
  delete (linear3_bias);
  delete (instanceNorm2d0_weight);
  delete (instanceNorm2d0_bias);
  delete (instanceNorm2d1_weight);
  delete (instanceNorm2d1_bias);
  delete (input);
  delete (output);
  delete (c1);
  delete (i1);
  delete (m1);
  delete (c2);
  delete (i2);
  delete (m2);
  delete (l1);
  delete (l2);
}
