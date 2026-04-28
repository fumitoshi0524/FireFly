#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

__device__ __forceinline__ float decode_code(uint8_t code)
{
    return code == 1 ? 1.0f : (code == 2 ? -1.0f : 0.0f);
}

__global__ void ff_packed_linear_forward_kernel(
    const float *x, const uint8_t *packed_w, float *out,
    int64_t M, int64_t N, int64_t K, int64_t B, float scale)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * N;
    if (idx >= total)
        return;
    int64_t m = idx / N;
    int64_t n = idx % N;

    const float *x_row = x + m * K;
    const uint8_t *w_row = packed_w + n * B;
    float acc = 0.0f;
    for (int64_t b = 0; b < B; ++b)
    {
        uint8_t packed = w_row[b];
        int64_t k0 = b * 4;
        if (k0 + 0 < K)
            acc += x_row[k0 + 0] * decode_code((packed >> 0) & 0x3);
        if (k0 + 1 < K)
            acc += x_row[k0 + 1] * decode_code((packed >> 2) & 0x3);
        if (k0 + 2 < K)
            acc += x_row[k0 + 2] * decode_code((packed >> 4) & 0x3);
        if (k0 + 3 < K)
            acc += x_row[k0 + 3] * decode_code((packed >> 6) & 0x3);
    }
    out[idx] = acc * scale;
}

__global__ void ff_packed_linear_backward_input_kernel(
    const float *grad_out, const uint8_t *packed_w, float *grad_x,
    int64_t M, int64_t N, int64_t K, int64_t B, float scale)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = M * K;
    if (idx >= total)
        return;
    int64_t m = idx / K;
    int64_t k = idx % K;
    int64_t b = k / 4;
    int lane = static_cast<int>(k % 4);
    int shift = lane * 2;

    float acc = 0.0f;
    for (int64_t n = 0; n < N; ++n)
    {
        const uint8_t packed = packed_w[n * B + b];
        const float w = decode_code((packed >> shift) & 0x3);
        acc += grad_out[m * N + n] * w;
    }
    grad_x[idx] = acc * scale;
}

__global__ void ff_packed_linear_backward_weight_kernel(
    const float *grad_out, const float *x, float *grad_w,
    int64_t M, int64_t N, int64_t K, float scale)
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
    grad_w[idx] = acc * scale;
}

torch::Tensor ff_packed_linear_forward_cuda(torch::Tensor x, torch::Tensor packed_w, int64_t in_features, double scale)
{
    auto x_c = x.contiguous();
    auto w_c = packed_w.contiguous();
    const auto M = x_c.size(0);
    const auto N = w_c.size(0);
    const auto B = w_c.size(1);
    auto out = torch::zeros({M, N}, x.options().dtype(torch::kFloat32));

    const int threads = 256;
    const int64_t total = M * N;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    ff_packed_linear_forward_kernel<<<blocks, threads>>>(
        x_c.data_ptr<float>(), w_c.data_ptr<uint8_t>(), out.data_ptr<float>(),
        M, N, in_features, B, static_cast<float>(scale));
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

torch::Tensor ff_packed_linear_backward_input_cuda(torch::Tensor grad_out, torch::Tensor packed_w, int64_t in_features, double scale)
{
    auto go_c = grad_out.contiguous();
    auto w_c = packed_w.contiguous();
    const auto M = go_c.size(0);
    const auto N = go_c.size(1);
    const auto B = w_c.size(1);
    auto grad_x = torch::zeros({M, in_features}, grad_out.options().dtype(torch::kFloat32));

    const int threads = 256;
    const int64_t total = M * in_features;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    ff_packed_linear_backward_input_kernel<<<blocks, threads>>>(
        go_c.data_ptr<float>(), w_c.data_ptr<uint8_t>(), grad_x.data_ptr<float>(),
        M, N, in_features, B, static_cast<float>(scale));
    C10_CUDA_CHECK(cudaGetLastError());
    return grad_x;
}

torch::Tensor ff_packed_linear_backward_weight_cuda(torch::Tensor grad_out, torch::Tensor x, double scale)
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
    ff_packed_linear_backward_weight_kernel<<<blocks, threads>>>(
        go_c.data_ptr<float>(), x_c.data_ptr<float>(), grad_w.data_ptr<float>(),
        M, N, K, static_cast<float>(scale));
    C10_CUDA_CHECK(cudaGetLastError());
    return grad_w;
}
