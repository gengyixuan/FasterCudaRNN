#include <torch/torch.h>

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> cache_lstm_forward(
    at::Tensor XU, // (T, B, 4Dh)
    at::Tensor W, // (4Dh, Dh)
    at::Tensor h0, // (1, B, Dh)
    at::Tensor c0  // (B, Dh)
);

std::vector<at::Tensor> cache_lstm_backward(
    at::Tensor grad_h,
    at::Tensor grad_cell,
    at::Tensor new_cell,
    at::Tensor input_gate,
    at::Tensor output_gate,
    at::Tensor candidate_cell,
    at::Tensor X,
    at::Tensor gate_weights,
    at::Tensor weights
);

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> lstm_forward(
    at::Tensor XU, // (T, B, 4Dh)
    at::Tensor W, // (4Dh, Dh)
    at::Tensor h0, // (1, B, Dh)
    at::Tensor c0  // (B, Dh)
) {
  CHECK_INPUT(XU);
  CHECK_INPUT(W);
  CHECK_INPUT(h0);
  CHECK_INPUT(c0);

  return cache_lstm_forward(XU, W, h0, c0);
}

std::vector<at::Tensor> lstm_backward(
    at::Tensor grad_h,
    at::Tensor grad_cell,
    at::Tensor new_cell,
    at::Tensor input_gate,
    at::Tensor output_gate,
    at::Tensor candidate_cell,
    at::Tensor X,
    at::Tensor gate_weights,
    at::Tensor weights) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(grad_cell);
  CHECK_INPUT(input_gate);
  CHECK_INPUT(output_gate);
  CHECK_INPUT(candidate_cell);
  CHECK_INPUT(X);
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(weights);

  return cache_lstm_backward(
      grad_h,
      grad_cell,
      new_cell,
      input_gate,
      output_gate,
      candidate_cell,
      X,
      gate_weights,
      weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lstm_forward, "LSTM forward (CUDA)");
  m.def("backward", &lstm_backward, "LSTM backward (CUDA)");
}