#include <torch/extension.h>

torch::Tensor ff_packed_linear_forward_cuda(torch::Tensor x, torch::Tensor packed_w, int64_t in_features, double scale);
torch::Tensor ff_packed_linear_backward_input_cuda(torch::Tensor grad_out, torch::Tensor packed_w, int64_t in_features, double scale);
torch::Tensor ff_packed_linear_backward_weight_cuda(torch::Tensor grad_out, torch::Tensor x, double scale);

#if !FIREFLY_USE_CUDA
torch::Tensor ff_packed_linear_forward_cuda(torch::Tensor, torch::Tensor, int64_t, double)
{
    TORCH_CHECK(false, "firefly_bitnet_ext was built without CUDA support");
}

torch::Tensor ff_packed_linear_backward_input_cuda(torch::Tensor, torch::Tensor, int64_t, double)
{
    TORCH_CHECK(false, "firefly_bitnet_ext was built without CUDA support");
}

torch::Tensor ff_packed_linear_backward_weight_cuda(torch::Tensor, torch::Tensor, double)
{
    TORCH_CHECK(false, "firefly_bitnet_ext was built without CUDA support");
}
#endif

static inline float decode_code(uint8_t code)
{
    return code == 1 ? 1.0f : (code == 2 ? -1.0f : 0.0f);
}

torch::Tensor ff_packed_linear_forward_cpu(torch::Tensor x, torch::Tensor packed_w, int64_t in_features, double scale)
{
    auto x_c = x.contiguous();
    auto w_c = packed_w.contiguous();
    const auto M = x_c.size(0);
    const auto N = w_c.size(0);
    const auto B = w_c.size(1);
    auto out = torch::zeros({M, N}, x.options().dtype(torch::kFloat32));

    const float *x_ptr = x_c.data_ptr<float>();
    const uint8_t *w_ptr = w_c.data_ptr<uint8_t>();
    float *out_ptr = out.data_ptr<float>();

    for (int64_t m = 0; m < M; ++m)
    {
        for (int64_t n = 0; n < N; ++n)
        {
            const float *x_row = x_ptr + m * in_features;
            const uint8_t *w_row = w_ptr + n * B;
            float acc = 0.0f;
            for (int64_t b = 0; b < B; ++b)
            {
                const uint8_t packed = w_row[b];
                const int64_t k0 = b * 4;
                if (k0 + 0 < in_features)
                    acc += x_row[k0 + 0] * decode_code((packed >> 0) & 0x3);
                if (k0 + 1 < in_features)
                    acc += x_row[k0 + 1] * decode_code((packed >> 2) & 0x3);
                if (k0 + 2 < in_features)
                    acc += x_row[k0 + 2] * decode_code((packed >> 4) & 0x3);
                if (k0 + 3 < in_features)
                    acc += x_row[k0 + 3] * decode_code((packed >> 6) & 0x3);
            }
            out_ptr[m * N + n] = acc * static_cast<float>(scale);
        }
    }
    return out;
}

torch::Tensor ff_packed_linear_backward_input_cpu(torch::Tensor grad_out, torch::Tensor packed_w, int64_t in_features, double scale)
{
    auto go_c = grad_out.contiguous();
    auto w_c = packed_w.contiguous();
    const auto M = go_c.size(0);
    const auto N = go_c.size(1);
    const auto B = w_c.size(1);
    auto grad_x = torch::zeros({M, in_features}, grad_out.options().dtype(torch::kFloat32));

    const float *go_ptr = go_c.data_ptr<float>();
    const uint8_t *w_ptr = w_c.data_ptr<uint8_t>();
    float *gx_ptr = grad_x.data_ptr<float>();

    for (int64_t m = 0; m < M; ++m)
    {
        float *gx_row = gx_ptr + m * in_features;
        for (int64_t n = 0; n < N; ++n)
        {
            const float go = go_ptr[m * N + n] * static_cast<float>(scale);
            const uint8_t *w_row = w_ptr + n * B;
            for (int64_t b = 0; b < B; ++b)
            {
                const uint8_t packed = w_row[b];
                const int64_t k0 = b * 4;
                if (k0 + 0 < in_features)
                    gx_row[k0 + 0] += go * decode_code((packed >> 0) & 0x3);
                if (k0 + 1 < in_features)
                    gx_row[k0 + 1] += go * decode_code((packed >> 2) & 0x3);
                if (k0 + 2 < in_features)
                    gx_row[k0 + 2] += go * decode_code((packed >> 4) & 0x3);
                if (k0 + 3 < in_features)
                    gx_row[k0 + 3] += go * decode_code((packed >> 6) & 0x3);
            }
        }
    }
    return grad_x;
}

torch::Tensor ff_packed_linear_backward_weight_cpu(torch::Tensor grad_out, torch::Tensor x, double scale)
{
    auto go_c = grad_out.contiguous();
    auto x_c = x.contiguous();
    auto grad_w = torch::matmul(go_c.transpose(0, 1), x_c);
    if (scale != 1.0)
        grad_w.mul_(static_cast<float>(scale));
    return grad_w;
}

torch::Tensor ff_packed_linear_forward(torch::Tensor x, torch::Tensor packed_w, int64_t in_features, double scale)
{
    if (x.is_cuda())
        return ff_packed_linear_forward_cuda(x, packed_w, in_features, scale);
    return ff_packed_linear_forward_cpu(x, packed_w, in_features, scale);
}

torch::Tensor ff_packed_linear_backward_input(torch::Tensor grad_out, torch::Tensor packed_w, int64_t in_features, double scale)
{
    if (grad_out.is_cuda())
        return ff_packed_linear_backward_input_cuda(grad_out, packed_w, in_features, scale);
    return ff_packed_linear_backward_input_cpu(grad_out, packed_w, in_features, scale);
}

torch::Tensor ff_packed_linear_backward_weight(torch::Tensor grad_out, torch::Tensor x, double scale)
{
    if (grad_out.is_cuda())
        return ff_packed_linear_backward_weight_cuda(grad_out, x, scale);
    return ff_packed_linear_backward_weight_cpu(grad_out, x, scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ff_packed_linear_forward", &ff_packed_linear_forward, "packed ternary linear forward");
    m.def("ff_packed_linear_backward_input", &ff_packed_linear_backward_input, "packed ternary linear backward input");
    m.def("ff_packed_linear_backward_weight", &ff_packed_linear_backward_weight, "packed ternary linear backward weight");
}
