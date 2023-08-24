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

static void linear_redu(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                        Tensor *bias_t, Tensor *redu_buf_t);

// ReLU (inplace)
// https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
// size of in & out = N
static void relu(Tensor *inout_t);

#define CHECK_TIME(time_value) \
  CHECK_CUDA(cudaDeviceSynchronize()); \
  end_t = get_time(); time_value += end_t - start_t; start_t = end_t; \

void model_forward(float *inputN, float *outputN) {
  int batch = std::min(N, MAX_BATCH_SIZE);
  int batch_count = (N + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE; 
  int count = 0;
  double conv2d_t = 0.0;
  double instance_t = 0.0;
  double maxpool2d_t = 0.0;
  double relu_t  = 0.0;
  double linear_t = 0.0;                 
  double memcp_t = 0.0;
  double start_t = get_time();
  double end_t;
  fprintf(stderr, "\ntotal count: %d\n", N);
  for (int idx = 0 ; idx < batch_count ; idx++) {
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
    CHECK_TIME(memcp_t);
    conv2d(input, c1, conv0_weight, conv0_bias);
    CHECK_TIME(conv2d_t);
    instancenorm2d(c1, i1, instanceNorm2d0_weight, instanceNorm2d0_bias);
    CHECK_TIME(instance_t);
    maxpool2d(i1, m1, 2, 2);
    CHECK_TIME(maxpool2d_t);
    relu(m1);
    CHECK_TIME(relu_t);
    conv2d(m1, c2, conv1_weight, conv1_bias);
    CHECK_TIME(conv2d_t);
    instancenorm2d(c2, i2, instanceNorm2d1_weight, instanceNorm2d1_bias);
    CHECK_TIME(instance_t);
    maxpool2d(i2, m2, 2, 2);
    CHECK_TIME(maxpool2d_t);
    relu(m2);
    CHECK_TIME(relu_t);
    linear(m2, l1, linear1_weight, linear1_bias);
    CHECK_TIME(linear_t);
    relu(l1);
    CHECK_TIME(relu_t);
    linear(l1, l2, linear2_weight, linear2_bias);
    l2->reshape({batch, 1, 1015808});
    linear_redu(l2, output, linear3_weight, linear3_bias, l1);
    CHECK_TIME(linear_t);
    CHECK_CUDA(cudaMemcpy(outputN, output->buf, 2 * sizeof(float) * batch,
                          cudaMemcpyDeviceToHost));
    CHECK_TIME(memcp_t);
    outputN += 2 * batch;
    count += batch;
    fprintf(stderr, "batch [%d/%d]...\n", count, N);
  }

  fprintf(stderr,"conv2d: %lf\n", conv2d_t);
  fprintf(stderr,"maxpool2d: %lf\n", maxpool2d_t);
  fprintf(stderr,"linear: %lf\n", linear_t);
  fprintf(stderr,"instance: %lf\n", instance_t);
  fprintf(stderr,"relu: %lf\n", relu_t);
  fprintf(stderr,"cudaMemcpy: %lf\n", memcp_t); 
}

__global__ void conv2d_kernel(float *in, float *out, float *weight, float *bias,
                              int C_IN, int C_OUT, int H_IN, int W_IN,
                              int H_OUT, int W_OUT, int B){
  int b = blockIdx.y / C_OUT;
  int c_out = blockIdx.y % C_OUT;
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  int h_out = tidx / W_OUT;
  int w_out = tidx % W_OUT;
  const int K = 3;
  __shared__ float shrWg[2048];

  float val = bias[c_out];
  if (c_out * C_IN * K * K + threadIdx.x < C_IN * C_OUT * K * K)
    shrWg[threadIdx.x] = weight[c_out * C_IN * K * K + threadIdx.x];
  else
    shrWg[threadIdx.x] = 0;

  if (C_IN==128 && threadIdx.x < C_IN) {
    if (c_out * C_IN * K * K + threadIdx.x + 1024 < C_IN * C_OUT * K * K)
      shrWg[1024 + threadIdx.x] = weight[c_out * C_IN * K * K + threadIdx.x + 1024];
    else
      shrWg[1024 + threadIdx.x] = 0;
  }
  __syncthreads();

  if (tidx >= H_OUT * W_OUT) return;
  out += b * C_OUT * H_OUT * W_OUT;
  in += b * C_IN * H_IN * W_IN;
  for (int c_in = 0; c_in < C_IN; c_in++) {
    for (int kh = 0; kh < K; kh++) {
      for (int kw = 0; kw < K; kw++) {
        val += in[c_in * H_IN * W_IN + (h_out + kh) * W_IN + (w_out + kw)] *
          shrWg[c_in * K * K + kh * K + kw];
      }
    }
  }
  out[c_out * H_OUT * W_OUT + h_out * W_OUT + w_out] = val;
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
  //  int n_thread = batch * C_OUT * H_OUT * W_OUTW;
  //  dim3 blockDim(1024);
  //dim3 gridDim((n_thread + 1023) / 1024);

  dim3 blockDim(1024);
  dim3 gridDim(((H_OUT * W_OUT) + 1023) / 1024, batch * C_OUT);
  conv2d_kernel<<<gridDim, blockDim>>>(in, out, weight, bias, C_IN, C_OUT, H_IN, W_IN, H_OUT, W_OUT, batch);
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

__global__ void instancenorm2d_kernel_reduc(float *in, float *out, float *weight, float *bias, int C, int H, int W, int B, float *E, float *V){
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = tidx / (C * H * W);
    int c = (tidx / (H * W)) % C;
    int h = (tidx / W) % H;
    int w = tidx % W;

    out[b * C * H * W + c * H * W + h * W + w] = (in[b * C * H * W + c * H * W + h * W + w] - E[b * C + c]) / sqrt(V[b * C + c] + 1e-5) * weight[c] + bias[c];
}


__global__ void calc_mean_variance_kernel(float *in, float *E, float *V, int C, int H, int W, int B){

    unsigned int tid = threadIdx.x;
    unsigned int R = blockIdx.x;
    unsigned int BLOCK_SIZE = blockDim.x; // n_thread (1024) 
    int n_array = H * W;

    //int size_shm = B * C * BLOCK_SIZE;
    //extern __shared__ float L[];
    __shared__ float L[1024];

    // calc mean 
    float e = 0.0;
    for (int i = tid; i < n_array; i+= BLOCK_SIZE){
	e += in[R * H * W + i];
    }
    L[tid] = e;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2){
	if (tid < stride) L[tid] += L[tid + stride];
	__syncthreads();
    }

    if (tid == 0) E[R] = L[tid] / (H * W);
    __syncthreads();

    // calc variance
    __shared__ float M[1024];

    float v = 0.0;
    for (int i = tid; i < n_array; i+=BLOCK_SIZE){
	v += (in[R * H * W + i] - E[R]) * (in[R * H * W + i] - E[R]);
    }

    M[tid] = v;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2){
	if (tid < stride) M[tid] += M[tid + stride];
	__syncthreads();
    }
    if (tid == 0) V[R] = M[tid] / (H * W);
    __syncthreads();
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

  float *E, *V;
  CHECK_CUDA(cudaMalloc(&E, sizeof(float) * B * C));
  CHECK_CUDA(cudaMalloc(&V, sizeof(float) * B * C));
  // kernel function to get E, V
  dim3 gridDim(B * C);
  dim3 blockDim(1024);
  calc_mean_variance_kernel<<<gridDim, blockDim>>>(in, E, V, C, H, W, B);

  //apply instancenorm with E, V

  int n_thread = B * C * H * W;
  dim3 block_dim(512);
  dim3 grid_dim((n_thread + 512 -1) / 512);

  instancenorm2d_kernel_reduc<<<grid_dim, block_dim>>>(in, out, weight, bias, C, H, W, B, E, V);
}

#define BLOCK_SIZE (32)

__global__ void linear_kernel(float *in, float *out, float* weight, float* bias,
                              int H_IN, int H_OUT, int N, int batch) {
  int b = blockIdx.z;
  int h_out = blockDim.x * blockIdx.x + threadIdx.x;
  int n = blockDim.y * blockIdx.y + threadIdx.y;

  __shared__ float shrIn[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float shrWg[BLOCK_SIZE][BLOCK_SIZE];

  float val = bias[h_out];
  for (int i = 0; i < (H_IN+BLOCK_SIZE-1)/BLOCK_SIZE; i++) {
    int offset = i * BLOCK_SIZE;

    if (n < N && threadIdx.x + offset < H_IN)
      shrIn[threadIdx.y][threadIdx.x] = in[b * N * H_IN + n * H_IN + offset + threadIdx.x];
    else
      shrIn[threadIdx.y][threadIdx.x] = 0;

    if (h_out < H_OUT && threadIdx.y + offset < H_IN)
      shrWg[threadIdx.y][threadIdx.x] = weight[h_out * H_IN + offset + threadIdx.y];
    else
      shrWg[threadIdx.y][threadIdx.x] = 0;

    __syncthreads();

    for (int j = 0 ; j <BLOCK_SIZE ; j++) {
      val += shrIn[threadIdx.y][j] * shrWg[j][threadIdx.x];
    }
    __syncthreads();
  }

  if (n >= N || h_out >= H_OUT) return;
  out[b * N * H_OUT + n * H_OUT + h_out] = val;
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
  // int n_thread = N * H_OUT * batch;
  //  dim3 blockDim(512);
  // dim3 gridDim((n_thread + 512 - 1) / 512);
  dim3 gridDim((H_OUT+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE, batch);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  linear_kernel<<<gridDim, blockDim>>>(in, out, weight, bias, H_IN, H_OUT, N, batch);
}

__global__ void linear_redu_kernel_1(float *in, float *out, float *weight, int H_IN) {
  int tidx = threadIdx.x;
  int h_in = blockDim.x * blockIdx.x + tidx;
  int b = blockIdx.z;
  const int H_OUT = 2;
  const int N = 1;
  const int n = 0;
  int h_out = blockIdx.y;
  int block_size = gridDim.x;
  __shared__ float L[1024];
  L[tidx] = 0;
  if (h_in >= H_IN) return;

  L[tidx] = in[b * N * H_IN + n * H_IN + h_in] * weight[h_out * H_IN + h_in];
  __syncthreads();
  for (int stride = 512 ; stride > 0 ; stride /= 2) {
    if (tidx < stride) L[tidx] += L[tidx + stride];
    __syncthreads();
  }

  if (tidx == 0) out[block_size * (b * H_OUT + h_out) + blockIdx.x] = L[0];
}

__global__ void linear_redu_kernel_2(float *in, float *out, float *bias, const int block_size) {
  const int tidx = threadIdx.x;
  const int h_out = blockIdx.y;
  const int b = blockIdx.z;
  const int H_OUT = 2;
  const int n = 0;
  const int N = 1;
  __shared__ float L[1024];
  L[tidx] = 0;
  if (tidx >= block_size) return;
  L[tidx] = in[block_size * (b * H_OUT + h_out) + tidx];
  __syncthreads();
  for (int stride = 512 ; stride > 0 ; stride /= 2) {
    if (tidx < stride) L[tidx] += L[tidx + stride];
    __syncthreads();
  }

  if (tidx == 0) out[b * N * H_OUT + n * H_OUT + h_out] = L[0] + bias[h_out];
}

static void linear_redu(Tensor *in_t, Tensor *out_t, Tensor *weight_t,
                        Tensor *bias_t, Tensor *redu_buf_t) {
  float *in = in_t->buf;
  float *out = out_t->buf;
  float *weight = weight_t->buf;
  float *bias = bias_t->buf;
  float *buf = redu_buf_t->buf;

  int batch = in_t->shape[0];
  int H_IN = weight_t->shape[0];  // in_t의 마지막 차원
  int H_OUT = weight_t->shape[1]; // out_t의 마지막 차원

  int N = in_t->get_elem() / H_IN / batch ; //=out_t->get_elem()/H_OUT
  // get_elem() already include batch
  dim3 blockDim(1024);
  dim3 gridDim((H_IN+1023)/1024, H_OUT, batch);
  linear_redu_kernel_1<<<gridDim, blockDim>>>(in, buf, weight, H_IN);
  int block_size = gridDim.x;
  dim3 gridDim2(1, H_OUT, batch);
  linear_redu_kernel_2<<<gridDim2, blockDim>>>(buf, out, bias, block_size);
}

__global__ void maxpool2d_kernel(float *in, float *out, 
	int H_IN, int W_IN, int H_OUT, int W_OUT, int N, int kH, int kW, int batch){
  int tidx = blockDim.x * blockIdx.x + threadIdx.x;

  int b = tidx / (N * H_OUT * W_OUT);
  int n = (tidx / (H_OUT * W_OUT)) % N;
  int h_out = (tidx / W_OUT) % H_OUT;
  int w_out = tidx % W_OUT;

  if (b >= batch || n >= N || h_out >= H_OUT || w_out >= W_OUT) return;

  float val = in[b * N * H_IN * W_IN + n * H_IN * W_IN + (h_out * kH) * H_IN + (w_out * kW)];
  for (int kh = 0; kh < kH; kh++)
    for (int kw = 0; kw < kW; kw++)
      val = fmaxf(val,
                in[b * N * H_IN * W_IN + n * H_IN * W_IN + (h_out * kH + kh) * H_IN +
                   (w_out * kW + kw)]);
  out[b * N * H_OUT * W_OUT + n * H_OUT * W_OUT + h_out * W_OUT + w_out] = val;
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
