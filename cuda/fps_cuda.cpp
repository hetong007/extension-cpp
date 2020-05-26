#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> fps_cuda_interface(
    torch::Tensor src,
    torch::Tensor ptr,
    double ratio,
    bool random_start);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> fps_cuda(
    torch::Tensor src,
    torch::Tensor ptr,
    double ratio,
    bool random_start) {}
  CHECK_INPUT(src);
  CHECK_INPUT(ptr);

  return fps_cuda_interface(src, ptr, ratio, random_start);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fps_cuda", &fps_cuda, "Farthest Point Sampling (CUDA)");
}
