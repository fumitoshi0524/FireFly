#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

__global__ void ff_int8_linear_forward_kernel(
    const int8_t* x_q,
    const int8_t* w_q,
    int32_t* out,
    int64_t M,
    int64_t N,
    int64_t K)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    if (idx >= total)
        return;
    int64_t m = idx / N;
    int64_t n = idx % N;
    int32_t acc = 0;
    const int8_t* x_row = x_q + m * K;
    const int8_t* w_row = w_q + n * K;
    for (int64_t k = 0; k < K; ++k)
        acc += static_cast<int32_t>(x_row[k]) * static_cast<int32_t>(w_row[k]);
    out[idx] = acc;
}

__global__ void ff_int8_linear_backward_input_kernel(
    const float* grad_out,
    const int8_t* w_q,
    const float* w_scale,
    const float* c_scale,
    float* grad_x,
    int64_t M,
    int64_t N,
    int64_t K)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * K;
    if (idx >= total)
        return;
    int64_t m = idx / K;
    int64_t k = idx % K;
    float acc = 0.0f;
    for (int64_t n = 0; n < N; ++n)
    {
        const float w = static_cast<float>(w_q[n * K + k]) * w_scale[n] * c_scale[n];
        acc += grad_out[m * N + n] * w;
    }
    grad_x[idx] = acc;
}

__global__ void ff_int8_linear_backward_weight_kernel(
    const float* grad_out,
    const float* x,
    float* grad_w,
    int64_t M,
    int64_t N,
    int64_t K)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * K;
    if (idx >= total)
        return;
    int64_t n = idx / K;
    int64_t k = idx % K;
    float acc = 0.0f;
    for (int64_t m = 0; m < M; ++m)
        acc += grad_out[m * N + n] * x[m * K + k];
    grad_w[idx] = acc;
}

torch::Tensor ff_int8_linear_forward_cuda(torch::Tensor x_q, torch::Tensor w_q)
{
    auto x_c = x_q.contiguous();
    auto w_c = w_q.contiguous();
    const auto M = x_c.size(0);
    const auto K = x_c.size(1);
    const auto N = w_c.size(0);
    auto out = torch::zeros({M, N}, x_q.options().dtype(torch::kInt32));

    const int threads = 256;
    const int64_t total = M * N;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    ff_int8_linear_forward_kernel<<<blocks, threads>>>(
        x_c.data_ptr<int8_t>(),
        w_c.data_ptr<int8_t>(),
        out.data_ptr<int32_t>(),
        M,
        N,
        K);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

torch::Tensor ff_int8_linear_backward_input_cuda(
    torch::Tensor grad_out,
    torch::Tensor w_q,
    torch::Tensor w_scale,
    torch::Tensor c_scale)
{
    auto go_c = grad_out.contiguous();
    auto wq_c = w_q.contiguous();
    auto ws_c = w_scale.contiguous();
    auto cs_c = c_scale.contiguous();
    const auto M = go_c.size(0);
    const auto N = go_c.size(1);
    const auto K = wq_c.size(1);
    auto grad_x = torch::zeros({M, K}, grad_out.options().dtype(torch::kFloat32));

    const int threads = 256;
    const int64_t total = M * K;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    ff_int8_linear_backward_input_kernel<<<blocks, threads>>>(
        go_c.data_ptr<float>(),
        wq_c.data_ptr<int8_t>(),
        ws_c.data_ptr<float>(),
        cs_c.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        M,
        N,
        K);
    C10_CUDA_CHECK(cudaGetLastError());
    return grad_x;
}

torch::Tensor ff_int8_linear_backward_weight_cuda(torch::Tensor grad_out, torch::Tensor x)
{
    auto go_c = grad_out.contiguous();
    auto x_c = x.contiguous();
    const auto M = go_c.size(0);
    const auto N = go_c.size(1);
    const auto K = x_c.size(1);
    auto grad_w = torch::zeros({N, K}, grad_out.options().dtype(torch::kFloat32));

    const int threads = 256;
    const int64_t total = N * K;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    ff_int8_linear_backward_weight_kernel<<<blocks, threads>>>(
        go_c.data_ptr<float>(),
        x_c.data_ptr<float>(),
        grad_w.data_ptr<float>(),
        M,
        N,
        K);
    C10_CUDA_CHECK(cudaGetLastError());
    return grad_w;
}
