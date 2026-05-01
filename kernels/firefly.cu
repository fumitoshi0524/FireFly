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
// ---------------------------------------------------------------------------

#define TILE_M 32   // rows per block  (matched to blockDim.x)
#define TILE_N 32   // cols per block  (each thread handles 4, blockDim.y = TILE_N/4)
#define TILE_K 64   // inner dim tile   (multiple of 4 for dp4a)

// ---- Forward:  C = A @ B^T   [M,K] @ [N,K]^T → [M,N] --------------------
__global__ void int8_gemm_forward_kernel(
    const int8_t* __restrict__ A,    // [M, K]  activations  (int8)
    const int8_t* __restrict__ B,    // [N, K]  weights      (int8)
    int32_t* __restrict__ C,         // [M, N]  output       (int32)
    int M, int N, int K)
{
    // One thread = 1 row × 4 columns of output
    // blockDim.x = TILE_M = 32,  blockDim.y = TILE_N / 4 = 8  → 256 threads
    const int row   = blockIdx.y * TILE_M + threadIdx.x;
    const int col0  = blockIdx.x * TILE_N + threadIdx.y * 4;

    // Register accumulator — 4 values along the N dimension
    int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

    const int K_blocks = (K + TILE_K - 1) / TILE_K;

    for (int kb = 0; kb < K_blocks; ++kb) {
        const int k0 = kb * TILE_K;
        const int k_limit = min(k0 + TILE_K, K);

        // Each thread walks its slice of K, loading 4-byte packs for dp4a
        for (int k = k0; k + 3 < k_limit; k += 4) {

            // Load 4 activation values as a packed uint32
            uint32_t a_pack;
            if (row < M) {
                const int off = row * K + k;
                a_pack = *reinterpret_cast<const uint32_t*>(A + off);
            } else {
                a_pack = 0;
            }

            // Load 4 weight values (B has N rows, we need 4 different rows)
            uint32_t b_pack[4];
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                int b_row = col0 + d;
                if (b_row < N && k < k_limit) {
                    const int off = b_row * K + k;
                    b_pack[d] = *reinterpret_cast<const uint32_t*>(B + off);
                } else {
                    b_pack[d] = 0;
                }
            }

            // Four __dp4a instructions
            acc0 = __dp4a(b_pack[0], a_pack, acc0);
            acc1 = __dp4a(b_pack[1], a_pack, acc1);
            acc2 = __dp4a(b_pack[2], a_pack, acc2);
            acc3 = __dp4a(b_pack[3], a_pack, acc3);
        }
    }

    // Write 4 output elements
    if (row < M) {
        if (col0 + 0 < N) C[row * N + col0 + 0] = acc0;
        if (col0 + 1 < N) C[row * N + col0 + 1] = acc1;
        if (col0 + 2 < N) C[row * N + col0 + 2] = acc2;
        if (col0 + 3 < N) C[row * N + col0 + 3] = acc3;
    }
}


// ---- Backward input:  C = A @ B   [M,O] @ [O,K] → [M,K] -----------------
//     A: quantised grad_out  [M, O]  int8    (O = out_features)
//     B: int8 weights        [O, K]  int8    (K = in_features)
__global__ void int8_gemm_bw_input_kernel(
    const int8_t* __restrict__ A,    // [M, O]
    const int8_t* __restrict__ B,    // [O, K]
    int32_t* __restrict__ C,         // [M, K]
    int M, int O, int K)
{
    const int row   = blockIdx.y * TILE_M + threadIdx.x;
    const int col0  = blockIdx.x * TILE_N + threadIdx.y * 4;

    int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

    const int O_blocks = (O + TILE_K - 1) / TILE_K;

    for (int ob = 0; ob < O_blocks; ++ob) {
        const int o0 = ob * TILE_K;
        const int o_limit = min(o0 + TILE_K, O);

        for (int o = o0; o + 3 < o_limit; o += 4) {

            // Load 4 values from A along the O dimension
            uint32_t a_pack;
            if (row < M) {
                a_pack = *reinterpret_cast<const uint32_t*>(A + row * O + o);
            } else {
                a_pack = 0;
            }

            // Load 4 values from B for each of 4 output columns
            uint32_t b_pack[4];
            #pragma unroll
            for (int d = 0; d < 4; ++d) {
                int k = col0 + d;
                if (k < K) {
                    b_pack[d] = *reinterpret_cast<const uint32_t*>(B + o * K + k);
                } else {
                    b_pack[d] = 0;
                }
            }

            acc0 = __dp4a(b_pack[0], a_pack, acc0);
            acc1 = __dp4a(b_pack[1], a_pack, acc1);
            acc2 = __dp4a(b_pack[2], a_pack, acc2);
            acc3 = __dp4a(b_pack[3], a_pack, acc3);
        }
    }

    if (row < M) {
        if (col0 + 0 < K) C[row * K + col0 + 0] = acc0;
        if (col0 + 1 < K) C[row * K + col0 + 1] = acc1;
        if (col0 + 2 < K) C[row * K + col0 + 2] = acc2;
        if (col0 + 3 < K) C[row * K + col0 + 3] = acc3;
    }
}


// ---- Backward weight:  C = A^T @ B   [M,O]^T @ [M,K] → [O,K] -------------
//     A: quantised per-column grad_out  [M, O]  int8
//     B: quantised activation x         [M, K]  int8
__global__ void int8_gemm_bw_weight_kernel(
    const int8_t* __restrict__ A,    // [M, O]
    const int8_t* __restrict__ B,    // [M, K]
    int32_t* __restrict__ C,         // [O, K]
    int M, int O, int K)
{
    // C[O,K] = sum_m(A[m,O]^T @ B[m,K]) = sum_m(A[m,O] * B[m,K])
    // TILE across O (rows of output) and K (cols of output)
    // Contract over M dimension
    const int row   = blockIdx.y * TILE_M + threadIdx.x;   // O dimension
    const int col0  = blockIdx.x * TILE_N + threadIdx.y * 4; // K dimension

    int32_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;

    for (int m = 0; m + 3 < M; m += 4) {

        // Load 4 A values across the M dimension for this O row
        uint32_t a_pack;
        if (row < O) {
            a_pack = *reinterpret_cast<const uint32_t*>(A + m * O + row);
        } else {
            a_pack = 0;
        }

        // Load 4 B values across the M dimension for each K column
        uint32_t b_pack[4];
        #pragma unroll
        for (int d = 0; d < 4; ++d) {
            int k = col0 + d;
            if (k < K) {
                b_pack[d] = *reinterpret_cast<const uint32_t*>(B + m * K + k);
            } else {
                b_pack[d] = 0;
            }
        }

        acc0 = __dp4a(b_pack[0], a_pack, acc0);
        acc1 = __dp4a(b_pack[1], a_pack, acc1);
        acc2 = __dp4a(b_pack[2], a_pack, acc2);
        acc3 = __dp4a(b_pack[3], a_pack, acc3);
    }

    if (row < O) {
        if (col0 + 0 < K) C[row * K + col0 + 0] = acc0;
        if (col0 + 1 < K) C[row * K + col0 + 1] = acc1;
        if (col0 + 2 < K) C[row * K + col0 + 2] = acc2;
        if (col0 + 3 < K) C[row * K + col0 + 3] = acc3;
    }
}


// ---- Public entry points ---------------------------------------------------

static dim3 get_block()       { return dim3(TILE_M, TILE_N / 4); }   // (32, 8) = 256

torch::Tensor ff_int8_linear_forward_cuda(torch::Tensor x_q, torch::Tensor w_q)
{
    auto x_c = x_q.contiguous();
    auto w_c = w_q.contiguous();
    const int M = (int)x_c.size(0);
    const int K = (int)x_c.size(1);
    const int N = (int)w_c.size(0);
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
    const int M = (int)go_c.size(0);
    const int O = (int)go_c.size(1);
    const int K = (int)w_c.size(1);
    TORCH_CHECK(w_c.size(0) == O, "w_q shape mismatch for backward_input");

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
    const int M = (int)go_c.size(0);   // batch * seq
    const int O = (int)go_c.size(1);   // out_features
    const int K = (int)x_c.size(1);    // in_features
    TORCH_CHECK(x_c.size(0) == M, "x_q shape mismatch for backward_weight");

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
