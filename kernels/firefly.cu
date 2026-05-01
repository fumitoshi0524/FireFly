#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// ---------------------------------------------------------------------------
// Tiled INT8 GEMM using __dp4a  (llama.cpp-style)
//
// __dp4a computes 4-way int8 dot-product in one instruction:
//   sum = a0*b0 + a1*b1 + a2*b2 + a3*b3 + acc
//
// Available on all GPUs since Pascal (SM 6.1).  No cuBLASLt dependency.
// No dequantisation — operates directly on int8 → int32.
// Uses memcpy for packed loads → works with unaligned addresses.
// ---------------------------------------------------------------------------

#define TILE_M 32
#define TILE_N 32
#define TILE_K 64

// Packed 4-byte load, safe for unaligned addresses
__device__ __forceinline__ unsigned int load_packed(const int8_t* p) {
    unsigned int v = 0;
    memcpy(&v, p, 4);
    return v;
}

// ---- Forward:  C = A @ B^T   [M,K] @ [N,K]^T → [M,N] --------------------
__global__ void int8_gemm_forward_kernel(
    const int8_t* __restrict__ A,    // [M, K]  activations
    const int8_t* __restrict__ B,    // [N, K]  weights
    int32_t* __restrict__ C,         // [M, N]  output
    int M, int N, int K)
{
    const int row   = blockIdx.y * TILE_M + threadIdx.x;
    const int col0  = blockIdx.x * TILE_N + threadIdx.y * 4;

    unsigned int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

    for (int kb = 0; kb * TILE_K < K; ++kb) {
        const int k0 = kb * TILE_K;
        const int k_end = min(k0 + TILE_K, K);

        for (int k = k0; k + 3 < k_end; k += 4) {

            unsigned int a_pack = 0;
            if (row < M) a_pack = load_packed(A + row * K + k);

            unsigned int b_pack[4] = {0, 0, 0, 0};
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                int b_row = col0 + d;
                if (b_row < N) b_pack[d] = load_packed(B + b_row * K + k);
            }

            acc0 = __dp4a(b_pack[0], a_pack, acc0);
            acc1 = __dp4a(b_pack[1], a_pack, acc1);
            acc2 = __dp4a(b_pack[2], a_pack, acc2);
            acc3 = __dp4a(b_pack[3], a_pack, acc3);
        }
    }

    if (row < M) {
        if (col0 + 0 < N) C[row * N + col0 + 0] = (int32_t)acc0;
        if (col0 + 1 < N) C[row * N + col0 + 1] = (int32_t)acc1;
        if (col0 + 2 < N) C[row * N + col0 + 2] = (int32_t)acc2;
        if (col0 + 3 < N) C[row * N + col0 + 3] = (int32_t)acc3;
    }
}


// ---- Backward input:  C = A @ B   [M,O] @ [O,K] → [M,K] -----------------
__global__ void int8_gemm_bw_input_kernel(
    const int8_t* __restrict__ A,    // [M, O]
    const int8_t* __restrict__ B,    // [O, K]
    int32_t* __restrict__ C,         // [M, K]
    int M, int O, int K)
{
    const int row   = blockIdx.y * TILE_M + threadIdx.x;
    const int col0  = blockIdx.x * TILE_N + threadIdx.y * 4;

    unsigned int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

    for (int ob = 0; ob * TILE_K < O; ++ob) {
        const int o0 = ob * TILE_K;
        const int o_end = min(o0 + TILE_K, O);

        for (int o = o0; o + 3 < o_end; o += 4) {

            unsigned int a_pack = 0;
            if (row < M) a_pack = load_packed(A + row * O + o);

            unsigned int b_pack[4] = {0, 0, 0, 0};
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                int k = col0 + d;
                if (k < K) b_pack[d] = load_packed(B + o * K + k);
            }

            acc0 = __dp4a(b_pack[0], a_pack, acc0);
            acc1 = __dp4a(b_pack[1], a_pack, acc1);
            acc2 = __dp4a(b_pack[2], a_pack, acc2);
            acc3 = __dp4a(b_pack[3], a_pack, acc3);
        }
    }

    if (row < M) {
        if (col0 + 0 < K) C[row * K + col0 + 0] = (int32_t)acc0;
        if (col0 + 1 < K) C[row * K + col0 + 1] = (int32_t)acc1;
        if (col0 + 2 < K) C[row * K + col0 + 2] = (int32_t)acc2;
        if (col0 + 3 < K) C[row * K + col0 + 3] = (int32_t)acc3;
    }
}


// ---- Backward weight:  C = A^T @ B   [M,O]^T @ [M,K] → [O,K] -------------
__global__ void int8_gemm_bw_weight_kernel(
    const int8_t* __restrict__ A,    // [M, O]
    const int8_t* __restrict__ B,    // [M, K]
    int32_t* __restrict__ C,         // [O, K]
    int M, int O, int K)
{
    const int row   = blockIdx.y * TILE_M + threadIdx.x;   // O dim
    const int col0  = blockIdx.x * TILE_N + threadIdx.y * 4; // K dim

    unsigned int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

    for (int m = 0; m + 3 < M; m += 4) {

        unsigned int a_pack = 0;
        if (row < O) a_pack = load_packed(A + m * O + row);

        unsigned int b_pack[4] = {0, 0, 0, 0};
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            int k = col0 + d;
            if (k < K) b_pack[d] = load_packed(B + m * K + k);
        }

        acc0 = __dp4a(b_pack[0], a_pack, acc0);
        acc1 = __dp4a(b_pack[1], a_pack, acc1);
        acc2 = __dp4a(b_pack[2], a_pack, acc2);
        acc3 = __dp4a(b_pack[3], a_pack, acc3);
    }

    if (row < O) {
        if (col0 + 0 < K) C[row * K + col0 + 0] = (int32_t)acc0;
        if (col0 + 1 < K) C[row * K + col0 + 1] = (int32_t)acc1;
        if (col0 + 2 < K) C[row * K + col0 + 2] = (int32_t)acc2;
        if (col0 + 3 < K) C[row * K + col0 + 3] = (int32_t)acc3;
    }
}


// ---- Entry points ----------------------------------------------------------

static dim3 get_block() { return dim3(TILE_M, TILE_N / 4); }

torch::Tensor ff_int8_linear_forward_cuda(torch::Tensor x_q, torch::Tensor w_q)
{
    auto x_c = x_q.contiguous();
    auto w_c = w_q.contiguous();
    const int M = (int)x_c.size(0), K = (int)x_c.size(1), N = (int)w_c.size(0);
    TORCH_CHECK(w_c.size(1) == K, "w_q shape mismatch");
    auto out = torch::empty({M, N}, torch::dtype(torch::kInt32).device(x_q.device()));
    dim3 block = get_block();
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    int8_gemm_forward_kernel<<<grid, block>>>(
        x_c.data_ptr<int8_t>(), w_c.data_ptr<int8_t>(),
        out.data_ptr<int32_t>(), M, N, K);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

torch::Tensor ff_int8_linear_backward_input_cuda(torch::Tensor go_q, torch::Tensor w_q)
{
    auto go_c = go_q.contiguous();
    auto w_c = w_q.contiguous();
    const int M = (int)go_c.size(0), O = (int)go_c.size(1), K = (int)w_c.size(1);
    TORCH_CHECK(w_c.size(0) == O, "w_q shape mismatch");
    auto out = torch::empty({M, K}, torch::dtype(torch::kInt32).device(go_q.device()));
    dim3 block = get_block();
    dim3 grid((K + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    int8_gemm_bw_input_kernel<<<grid, block>>>(
        go_c.data_ptr<int8_t>(), w_c.data_ptr<int8_t>(),
        out.data_ptr<int32_t>(), M, O, K);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}

torch::Tensor ff_int8_linear_backward_weight_cuda(torch::Tensor go_pc_q, torch::Tensor x_q)
{
    auto go_c = go_pc_q.contiguous();
    auto x_c = x_q.contiguous();
    const int M = (int)go_c.size(0), O = (int)go_c.size(1), K = (int)x_c.size(1);
    TORCH_CHECK(x_c.size(0) == M, "x_q shape mismatch");
    auto out = torch::empty({O, K}, torch::dtype(torch::kInt32).device(go_pc_q.device()));
    dim3 block = get_block();
    dim3 grid((K + TILE_N - 1) / TILE_N, (O + TILE_M - 1) / TILE_M);
    int8_gemm_bw_weight_kernel<<<grid, block>>>(
        go_c.data_ptr<int8_t>(), x_c.data_ptr<int8_t>(),
        out.data_ptr<int32_t>(), M, O, K);
    C10_CUDA_CHECK(cudaGetLastError());
    return out;
}


// ---- INT8 weight update ----------------------------------------------------

__global__ void ff_int8_weight_update_kernel(
    int8_t* int_weight, const int32_t* delta_q, int64_t n)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t v = static_cast<int32_t>(int_weight[i]) + delta_q[i];
    if (v > 127)  v = 127;
    if (v < -127) v = -127;
    int_weight[i] = static_cast<int8_t>(v);
}

void ff_int8_weight_update_cuda(torch::Tensor int_weight, torch::Tensor delta_q)
{
    auto w_c = int_weight.contiguous();
    auto d_c = delta_q.contiguous();
    const int64_t n = w_c.numel();
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    ff_int8_weight_update_kernel<<<blocks, threads>>>(
        w_c.data_ptr<int8_t>(), d_c.data_ptr<int32_t>(), n);
    C10_CUDA_CHECK(cudaGetLastError());
    int_weight.copy_(w_c);
}
