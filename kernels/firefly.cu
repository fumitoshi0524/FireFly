#include <torch/extension.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// ---------------------------------------------------------------------------
// cuBLASLt handle — one per process, lazy-init, thread-safe via CUDA context
// ---------------------------------------------------------------------------
static cublasLtHandle_t get_lt_handle() {
    static cublasLtHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasStatus_t st = cublasLtCreate(&handle);
        TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                    "cublasLtCreate failed: ", (int)st);
    }
    return handle;
}

// ---------------------------------------------------------------------------
// Generic INT8 → INT32 GEMM via cuBLASLt  (bitsandbytes-style)
//
//   C  =  A^T  @  B          (cuBLAS column-major convention)
//
//   A  –  weights          [N, K]  row-major int8   (cuBLAS sees [K, N])
//   B  –  activations      [M, K]  row-major int8   (cuBLAS sees [K, M])
//   C  –  result           [M, N]  row-major int32  (cuBLAS sees [N, M])
//
// cuBLASLt automatically selects the best INT8 Tensor-Core kernel
// (Turing SM 7.5+, Ampere SM 8.0+, Ada SM 8.9).
// ---------------------------------------------------------------------------
static torch::Tensor int8_gemm_cublas(
    const torch::Tensor &A,     // [N, K] int8  (weights)
    const torch::Tensor &B)     // [M, K] int8  (activations)
{
    const int64_t N = A.size(0);   // out_features
    const int64_t K = A.size(1);   // in_features
    const int64_t M = B.size(0);   // batch * seq_len

    // Pad dimensions to multiples of 4  (cuBLAS INT8 requirement)
    const int64_t M_pad = (M + 3) / 4 * 4;
    const int64_t N_pad = (N + 3) / 4 * 4;

    auto A_c = A.contiguous();
    auto B_c = B.contiguous();

    torch::Tensor C;
    bool padded = (M != M_pad || N != N_pad);
    if (padded) {
        C = torch::zeros({M_pad, N_pad},
                         torch::dtype(torch::kInt32).device(A.device()));
    } else {
        C = torch::empty({M, N},
                         torch::dtype(torch::kInt32).device(A.device()));
    }

    auto handle = get_lt_handle();
    cublasLtMatrixLayout_t A_desc = nullptr, B_desc = nullptr, C_desc = nullptr;
    cublasLtMatmulDesc_t   matmul_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    // Layout descriptors  (cuBLAS sees row-major as column-major with swapped dims)
    cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_8I,
                               (int)N_pad, (int)K, (int)N_pad);   // [N,K] col-major
    cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_8I,
                               (int)K, (int)M_pad, (int)K);       // [K,M] col-major
    cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_32I,
                               (int)N_pad, (int)M_pad, (int)N_pad); // [N,M] col-major

    cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    cublasOperation_t trans_a = CUBLAS_OP_T;   // A: [K,N] → [N,K]
    cublasOperation_t trans_b = CUBLAS_OP_N;   // B: [K,M] stays
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &trans_a, sizeof(trans_a));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &trans_b, sizeof(trans_b));

    // Heuristic search
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws_size = 4 * 1024 * 1024UL;  // 4 MiB
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size));

    int returned = 0;
    cublasLtMatmulHeuristicResult_t result = {};
    cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc, A_desc, B_desc,
                                   C_desc, C_desc, pref, 1, &result, &returned);
    TORCH_CHECK(returned > 0, "cublasLtMatmulAlgoGetHeuristic returned 0 results");

    // Workspace allocation
    auto ws = torch::empty({(int64_t)ws_size},
                           torch::dtype(torch::kByte).device(A.device()));

    int32_t alpha = 1, beta = 0;
    cublasStatus_t st = cublasLtMatmul(
        handle, matmul_desc,
        &alpha,
        A_c.data_ptr<int8_t>(),  A_desc,
        B_c.data_ptr<int8_t>(),  B_desc,
        &beta,
        C.data_ptr<int32_t>(),   C_desc,
        C.data_ptr<int32_t>(),   C_desc,
        &result.algo,
        ws.data_ptr(), ws_size,
        0);  // default stream
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                "cublasLtMatmul failed: ", (int)st);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(matmul_desc);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(C_desc);

    if (padded) {
        return C.slice(0, 0, M).slice(1, 0, N).contiguous();
    }
    return C;
}

// ---------------------------------------------------------------------------
// Public entry points  (called from Python)
// ---------------------------------------------------------------------------

// Forward:  C  =  x_q  @  W_q^T     [M,K] @ [K,N] → [M,N] int32
torch::Tensor ff_int8_linear_forward_cuda(torch::Tensor x_q, torch::Tensor w_q)
{
    // int8_gemm_cublas(A=W_q, B=x_q) gives [M,N] from [N,K]^T @ [M,K]
    // That formula is: C = W_q^T @ x_q  →  we want x_q @ W_q^T
    // cuBLAS column-major: C = A^T @ B where A=W_q[N,K], B=x_q[M,K]
    //   = op(A) @ op(B) = A^T @ B = [N,K] row→[K,N] col, trans→[N,K] col
    //     @ [M,K] row→[K,M] col, no-trans→[K,M] col
    //   = [N,K] @ [K,M] = [N,M] col-major = [M,N] row-major  ✓
    return int8_gemm_cublas(w_q, x_q);
}

// Backward input:  grad_x  =  grad_out  @  W_q     [M,N] @ [N,K] → [M,K]
//   cuBLAS: result = int8_w_q^T @ grad_out → [K,N] @ [N,M] = [K,M] col = [M,K] row
//   But we need W_q @ grad_out^T ... let's use: grad_x = (W_q^T @ grad_out^T)^T
//   Simpler: grad_x = int8_gemm(A=grad_out[M,N], B=W_q[N,K])
//   C = A^T @ B = grad_out^T @ W_q = [N,M] @ [N,K]? No, dims don't match.
//
//   Let's think again. grad_x = grad_out @ W_q  in row-major.
//   grad_out: [M, N], W_q: [N, K], result: [M, K].
//   In cuBLAS col-major: grad_out_col = [N, M], W_q_col = [K, N].
//   We want result_col = [K, M].
//   C = W_q_col^T @ grad_out_col = [N, K] @ [N, M] = [K, M] col-major = [M, K] row ✓
//   So: A = W_q[N,K], B = grad_out[M,N] → same as forward but swap x_q for grad_out.
//   Wait, int8_gemm_cublas(A, B) gives C = A^T @ B = W_q^T @ grad_out = [N,K]^T @ [N,M].
//   But grad_out is [M, N], and our gemm expects B as [M, K] with K=in_features.
//   Here grad_out is [M, N] where N=out_features, not in_features.
//
//   Actually we need a DIFFERENT cuBLAS call. Let's add a variant.
torch::Tensor ff_int8_linear_backward_input_cuda(
    torch::Tensor grad_out,  // [M, N]  int8   (quantised grad output)
    torch::Tensor w_q)       // [N, K]  int8   (weights)
{
    const int64_t M = grad_out.size(0);
    const int64_t N = grad_out.size(1);   // out_features
    const int64_t K = w_q.size(1);        // in_features

    // grad_x = grad_out @ W_q   →  [M,N] @ [N,K] = [M,K]
    //
    // cuBLAS column-major:
    //   grad_out_col = [N, M], W_q_col = [K, N]
    //   C = W_q_col^T @ grad_out_col = [N,K] @ [N,M] = [K,M] col = [M,K] row  ✓
    //
    // So we need: A=W_q [N,K] (transposed), B=grad_out [M,N] (not transposed)
    // int8_gemm_cublas(W_q, grad_out) would give:
    //   C = W_q^T @ grad_out = [N,K]^T @ [M,N]  — but dims don't match.
    //
    // We need a separate path. Let's just do it inline.

    const int64_t M_pad = (M + 3) / 4 * 4;
    const int64_t K_pad = (K + 3) / 4 * 4;

    auto go_c = grad_out.contiguous();
    auto wq_c = w_q.contiguous();

    torch::Tensor C;
    bool padded = (M != M_pad || K != K_pad);
    if (padded) {
        C = torch::zeros({M_pad, K_pad},
                         torch::dtype(torch::kInt32).device(grad_out.device()));
    } else {
        C = torch::empty({M, K},
                         torch::dtype(torch::kInt32).device(grad_out.device()));
    }

    auto handle = get_lt_handle();
    cublasLtMatrixLayout_t A_desc = nullptr, B_desc = nullptr, C_desc = nullptr;
    cublasLtMatmulDesc_t   matmul_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    // A = W_q: [N,K] row-major → [K,N] col-major, transposed → [N,K] col-major
    cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_8I, (int)N, (int)K, (int)N);
    // B = grad_out: [M,N] row-major → [N,M] col-major, no trans
    cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_8I, (int)N, (int)M_pad, (int)N);
    // C = [K,M] col-major = [M,K] row-major
    cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_32I, (int)K_pad, (int)M_pad, (int)K_pad);

    cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    cublasOperation_t trans_a = CUBLAS_OP_T;
    cublasOperation_t trans_b = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &trans_a, sizeof(trans_a));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &trans_b, sizeof(trans_b));

    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws_size = 4 * 1024 * 1024UL;
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size));

    int returned = 0;
    cublasLtMatmulHeuristicResult_t result = {};
    cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc, A_desc, B_desc,
                                   C_desc, C_desc, pref, 1, &result, &returned);
    TORCH_CHECK(returned > 0,
                "backward_input: cublasLtMatmulAlgoGetHeuristic returned 0 results");

    auto ws = torch::empty({(int64_t)ws_size},
                           torch::dtype(torch::kByte).device(grad_out.device()));

    int32_t alpha = 1, beta = 0;
    cublasStatus_t st = cublasLtMatmul(
        handle, matmul_desc,
        &alpha,
        wq_c.data_ptr<int8_t>(),   A_desc,
        go_c.data_ptr<int8_t>(),   B_desc,
        &beta,
        C.data_ptr<int32_t>(),     C_desc,
        C.data_ptr<int32_t>(),     C_desc,
        &result.algo,
        ws.data_ptr(), ws_size, 0);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                "backward_input cublasLtMatmul failed: ", (int)st);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(matmul_desc);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(C_desc);

    if (padded) {
        return C.slice(0, 0, M).slice(1, 0, K).contiguous();
    }
    return C;
}

// Backward weight:  grad_w  =  go^T  @  x_q     [N,M] @ [M,K] → [N,K]
//   go: [M,N] int8   x_q: [M,K] int8   result: [N,K] int32
torch::Tensor ff_int8_linear_backward_weight_cuda(
    torch::Tensor go,   // [M, N] int8  (quantised grad_out, per-column quantised)
    torch::Tensor x_q)  // [M, K] int8  (quantised input activation)
{
    const int64_t M = go.size(0);
    const int64_t N = go.size(1);
    const int64_t K = x_q.size(1);

    // cuBLAS: C = A^T @ B
    // A = go: [M,N] row → [N,M] col, transposed → [M,N] col
    // B = x_q: [M,K] row → [K,M] col, no trans
    // C = A^T @ B = [M,N] @ [K,M]  — dims don't match: N vs K
    //
    // We need go^T @ x_q = [N,M] @ [M,K] = [N,K]
    //
    // cuBLAS: C = go_col @ x_q_col^T = [N,M] @ [M,K] = [N,K] col-major = [K,N] row.
    // TRANSA=N (no trans), TRANSB=T (transpose B)
    //
    // So: A=go[M,N], B=x_q[M,K]
    // cuBLAS A_col=[N,M] no trans, B_col=[K,M] transposed→[M,K]
    // C = [N,M] @ [M,K] = [N,K] col = [K,N] row
    // Wait, we want [N,K] row-major. In cuBLAS, [N,K] col = [K,N] row. Not right.
    //
    // Let me try transposing the result.
    // Or: C = B^T @ A = x_q_col^T @ go_col = [K,M]^T @ [N,M] = [M,K] @ [N,M]
    //   = [K,N] col = [N,K] row ✓
    //
    // So swap A and B:
    // A = x_q: [M,K] row → [K,M] col, TRANSA=N
    // B = go:   [M,N] row → [N,M] col, TRANSB=T → [M,N] col
    // C = A @ B = [K,M] @ [M,N] = [K,N] col = [N,K] row ✓

    const int64_t N_pad = (N + 3) / 4 * 4;
    const int64_t K_pad = (K + 3) / 4 * 4;

    auto go_c = go.contiguous();
    auto xq_c = x_q.contiguous();

    torch::Tensor C;
    bool padded = (N != N_pad || K != K_pad);
    if (padded) {
        C = torch::zeros({N_pad, K_pad},
                         torch::dtype(torch::kInt32).device(go.device()));
    } else {
        C = torch::empty({N, K},
                         torch::dtype(torch::kInt32).device(go.device()));
    }

    auto handle = get_lt_handle();
    cublasLtMatrixLayout_t A_desc = nullptr, B_desc = nullptr, C_desc = nullptr;
    cublasLtMatmulDesc_t   matmul_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    // A = x_q: [M,K] row → [K,M] col, no trans → stays [K,M]
    cublasLtMatrixLayoutCreate(&A_desc, CUDA_R_8I, (int)K_pad, (int)M, (int)K_pad);
    // B = go:   [M,N] row → [N,M] col, transposed → [M,N] col
    cublasLtMatrixLayoutCreate(&B_desc, CUDA_R_8I, (int)N_pad, (int)M, (int)N_pad);
    // C = [K,N] col = [N,K] row
    cublasLtMatrixLayoutCreate(&C_desc, CUDA_R_32I, (int)K_pad, (int)N_pad, (int)K_pad);

    cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32I, CUDA_R_32I);
    cublasOperation_t trans_a = CUBLAS_OP_N;
    cublasOperation_t trans_b = CUBLAS_OP_T;
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &trans_a, sizeof(trans_a));
    cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &trans_b, sizeof(trans_b));

    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws_size = 4 * 1024 * 1024UL;
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size));

    int returned = 0;
    cublasLtMatmulHeuristicResult_t result = {};
    cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc, A_desc, B_desc,
                                   C_desc, C_desc, pref, 1, &result, &returned);
    TORCH_CHECK(returned > 0,
                "backward_weight: cublasLtMatmulAlgoGetHeuristic returned 0 results");

    auto ws = torch::empty({(int64_t)ws_size},
                           torch::dtype(torch::kByte).device(go.device()));

    int32_t alpha = 1, beta = 0;
    cublasStatus_t st = cublasLtMatmul(
        handle, matmul_desc,
        &alpha,
        xq_c.data_ptr<int8_t>(),   A_desc,
        go_c.data_ptr<int8_t>(),   B_desc,
        &beta,
        C.data_ptr<int32_t>(),     C_desc,
        C.data_ptr<int32_t>(),     C_desc,
        &result.algo,
        ws.data_ptr(), ws_size, 0);
    TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS,
                "backward_weight cublasLtMatmul failed: ", (int)st);

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatmulDescDestroy(matmul_desc);
    cublasLtMatrixLayoutDestroy(A_desc);
    cublasLtMatrixLayoutDestroy(B_desc);
    cublasLtMatrixLayoutDestroy(C_desc);

    if (padded) {
        return C.slice(0, 0, N).slice(1, 0, K).contiguous();
    }
    // Transpose: cuBLAS gives [K,N] col = [N,K] row, but we store as [N,K]
    // Actually the C layout above gives row-major [N,K] directly because
    // we created the C descriptor with rows=K, cols=N, ld=K
    // In col-major: C has K rows, N cols → in row-major this is [N, K] ✓
    return C;
}

// ---------------------------------------------------------------------------
// INT8 weight update  (kept as a simple element-wise kernel — no cuBLAS needed)
// ---------------------------------------------------------------------------
__global__ void ff_int8_weight_update_kernel(
    int8_t* int_weight,
    const int32_t* delta_q,
    int64_t n)
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
        w_c.data_ptr<int8_t>(),
        d_c.data_ptr<int32_t>(),
        n);
    C10_CUDA_CHECK(cudaGetLastError());
    int_weight.copy_(w_c);
}
