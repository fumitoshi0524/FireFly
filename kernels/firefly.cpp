#include <torch/extension.h>

torch::Tensor ff_int8_linear_forward_cuda(torch::Tensor x_q, torch::Tensor w_q);
torch::Tensor ff_int8_linear_backward_input_cuda(torch::Tensor grad_out, torch::Tensor w_q, torch::Tensor w_scale, torch::Tensor c_scale);
torch::Tensor ff_int8_linear_backward_weight_cuda(torch::Tensor grad_out, torch::Tensor x);
void ff_int8_weight_update_cuda(torch::Tensor int_weight, torch::Tensor delta_q);

#if !FIREFLY_USE_CUDA
torch::Tensor ff_int8_linear_forward_cuda(torch::Tensor, torch::Tensor)
{
    TORCH_CHECK(false, "firefly_bitnet_ext was built without CUDA support");
}
torch::Tensor ff_int8_linear_backward_input_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor)
{
    TORCH_CHECK(false, "firefly_bitnet_ext was built without CUDA support");
}
torch::Tensor ff_int8_linear_backward_weight_cuda(torch::Tensor, torch::Tensor)
{
    TORCH_CHECK(false, "firefly_bitnet_ext was built without CUDA support");
}
void ff_int8_weight_update_cuda(torch::Tensor, torch::Tensor)
{
    TORCH_CHECK(false, "firefly_bitnet_ext was built without CUDA support");
}
#endif

torch::Tensor ff_int8_linear_forward_cpu(torch::Tensor x_q, torch::Tensor w_q)
{
    auto x_c = x_q.contiguous();
    auto w_c = w_q.contiguous();
    const auto M = x_c.size(0);
    const auto K = x_c.size(1);
    const auto N = w_c.size(0);
    TORCH_CHECK(w_c.size(1) == K, "w_q shape mismatch for int8 linear forward");

    auto out = torch::zeros({M, N}, x_q.options().dtype(torch::kInt32));
    const int8_t* x_ptr = x_c.data_ptr<int8_t>();
    const int8_t* w_ptr = w_c.data_ptr<int8_t>();
    int32_t* out_ptr = out.data_ptr<int32_t>();

    for (int64_t m = 0; m < M; ++m)
    {
        for (int64_t n = 0; n < N; ++n)
        {
            int32_t acc = 0;
            const int8_t* x_row = x_ptr + m * K;
            const int8_t* w_row = w_ptr + n * K;
            for (int64_t k = 0; k < K; ++k)
                acc += static_cast<int32_t>(x_row[k]) * static_cast<int32_t>(w_row[k]);
            out_ptr[m * N + n] = acc;
        }
    }
    return out;
}

torch::Tensor ff_int8_linear_backward_input_cpu(
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
    TORCH_CHECK(wq_c.size(0) == N, "w_q shape mismatch for int8 backward input");
    TORCH_CHECK(ws_c.size(0) == N && cs_c.size(0) == N, "scale shape mismatch for int8 backward input");

    auto grad_x = torch::zeros({M, K}, grad_out.options().dtype(torch::kFloat32));
    const float* go_ptr = go_c.data_ptr<float>();
    const int8_t* w_ptr = wq_c.data_ptr<int8_t>();
    const float* ws_ptr = ws_c.data_ptr<float>();
    const float* cs_ptr = cs_c.data_ptr<float>();
    float* gx_ptr = grad_x.data_ptr<float>();

    for (int64_t m = 0; m < M; ++m)
    {
        for (int64_t k = 0; k < K; ++k)
        {
            float acc = 0.0f;
            for (int64_t n = 0; n < N; ++n)
            {
                const float w = static_cast<float>(w_ptr[n * K + k]) * ws_ptr[n] * cs_ptr[n];
                acc += go_ptr[m * N + n] * w;
            }
            gx_ptr[m * K + k] = acc;
        }
    }
    return grad_x;
}

torch::Tensor ff_int8_linear_backward_weight_cpu(torch::Tensor grad_out, torch::Tensor x)
{
    auto go_c = grad_out.contiguous();
    auto x_c = x.contiguous();
    return torch::matmul(go_c.transpose(0, 1), x_c);
}

void ff_int8_weight_update_cpu(torch::Tensor int_weight, torch::Tensor delta_q)
{
    auto w_c = int_weight.contiguous();
    auto d_c = delta_q.contiguous();
    TORCH_CHECK(w_c.scalar_type() == torch::kInt8, "int_weight must be int8");
    TORCH_CHECK(d_c.scalar_type() == torch::kInt32, "delta_q must be int32");
    TORCH_CHECK(w_c.sizes() == d_c.sizes(), "shape mismatch in ff_int8_weight_update_");

    int8_t* w_ptr = w_c.data_ptr<int8_t>();
    const int32_t* d_ptr = d_c.data_ptr<int32_t>();
    const int64_t n = w_c.numel();
    for (int64_t i = 0; i < n; ++i)
    {
        int32_t v = static_cast<int32_t>(w_ptr[i]) + d_ptr[i];
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        w_ptr[i] = static_cast<int8_t>(v);
    }
    int_weight.copy_(w_c);
}

torch::Tensor ff_int8_linear_forward(torch::Tensor x_q, torch::Tensor w_q)
{
    if (x_q.is_cuda())
        return ff_int8_linear_forward_cuda(x_q, w_q);
    return ff_int8_linear_forward_cpu(x_q, w_q);
}

torch::Tensor ff_int8_linear_backward_input(
    torch::Tensor grad_out,
    torch::Tensor w_q,
    torch::Tensor w_scale,
    torch::Tensor c_scale)
{
    if (grad_out.is_cuda())
        return ff_int8_linear_backward_input_cuda(grad_out, w_q, w_scale, c_scale);
    return ff_int8_linear_backward_input_cpu(grad_out, w_q, w_scale, c_scale);
}

torch::Tensor ff_int8_linear_backward_weight(torch::Tensor grad_out, torch::Tensor x)
{
    if (grad_out.is_cuda())
        return ff_int8_linear_backward_weight_cuda(grad_out, x);
    return ff_int8_linear_backward_weight_cpu(grad_out, x);
}

void ff_int8_weight_update_(torch::Tensor int_weight, torch::Tensor delta_q)
{
    if (int_weight.is_cuda())
    {
        ff_int8_weight_update_cuda(int_weight, delta_q);
        return;
    }
    ff_int8_weight_update_cpu(int_weight, delta_q);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ff_int8_linear_forward", &ff_int8_linear_forward, "int8 linear forward (int8xint8->int32)");
    m.def("ff_int8_linear_backward_input", &ff_int8_linear_backward_input, "int8 linear backward input");
    m.def("ff_int8_linear_backward_weight", &ff_int8_linear_backward_weight, "int8 linear backward weight");
    m.def("ff_int8_weight_update_", &ff_int8_weight_update_, "int8 in-place weight update");
}
