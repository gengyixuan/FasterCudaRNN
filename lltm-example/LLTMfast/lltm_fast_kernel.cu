#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmax(0.0, z) + fmin(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}


// ===========================================
// Forward
// ===========================================


template <typename scalar_t>
__global__ void lltm_fast_forward_kernel(
    const scalar_t* __restrict__ X,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ gates,
    const scalar_t* __restrict__ old_cell,
    scalar_t* __restrict__ new_h,
    scalar_t* __restrict__ new_cell,
    scalar_t* __restrict__ input_gate,
    scalar_t* __restrict__ output_gate,
    scalar_t* __restrict__ candidate_cell,
    size_t X_width,
    size_t state_dim) {
  const int batch_id = blockIdx.y;
  const int column = threadIdx.x;
  const int index = batch_id * state_dim + column;
  const int gates_row = batch_id * (state_dim * 3);

  // X: (batch, state_dim + input_dim)
  // weights: (state_dim * 3, state_dim + input_dim)
  // bias: (state_dim * 3)

  if (column < state_dim) {
    int i;
    int X_row_start = batch_id * X_width;
    
    // ==================================================================================
    // linear layer: [h x] * W + b 
    //            -> input_gate, output_gate, candidate_cell
    
    // input_gate[batch, col] = sum_over_i(X[batch, i] * weights[col, i]) + bias[col]
    int input_gate_index_sum = 0;
    int col_input = column;
    int weights_row_start_input = col_input * X_width;
    // output_gate[batch, col] = sum_over_i(X[batch, i] * weights[state_dim + col, i]) + bias[state_dim + col]
    int output_gate_index_sum = 0;
    int col_output = state_dim + column;
    int weights_row_start_output = col_output * X_width;
    // candidate_cell[batch, col] = sum_over_i(X[batch, i] * weights[2 * state_dim + col, i]) + bias[2 * state_dim + col]
    int cand_gate_index_sum = 0;
    int col_cand = 2 * state_dim + column;
    int weights_row_start_cand = col_cand * X_width;

    // dot product for 3 pairs of vectors
    for (i = 0; i < X_width; i++) {
      int xx = X[X_row_start + i];
      input_gate_index_sum += xx * weights[weights_row_start_input + i];
      output_gate_index_sum += xx * weights[weights_row_start_output + i];
      cand_gate_index_sum += xx * weights[weights_row_start_cand + i];
    }
    gates[gates_row + column] = input_gate_index_sum + bias[col_input];
    gates[gates_row + state_dim + column] = output_gate_index_sum + bias[col_output];
    gates[gates_row + 2 * state_dim + column] = cand_gate_index_sum + bias[col_cand];   

    // ===================================================================================
    // activation layer
    input_gate[index] = sigmoid(gates[gates_row + column]);
    output_gate[index] = sigmoid(gates[gates_row + state_dim + column]);
    candidate_cell[index] = elu(gates[gates_row + 2 * state_dim + column]);

    // ===================================================================================
    // output layer
    new_cell[index] = old_cell[index] + candidate_cell[index] * input_gate[index];
    new_h[index] = tanh(new_cell[index]) * output_gate[index];
  }
}


std::vector<at::Tensor> lltm_fast_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell) {
  // X should be on gpu already
  auto X = at::cat({old_h, input}, /*dim=*/1);

  const auto batch_size = old_cell.size(0);
  const auto state_dim = old_cell.size(1);
  const auto input_dim = input.size(1); // input: (batch, input_dim, input_len)
  //const auto input_len = input.size(2); // input: (batch, input_dim, input_len)

  auto new_h = at::zeros_like(old_cell);
  auto new_cell = at::zeros_like(old_cell);
  auto input_gate = at::zeros_like(old_cell);
  auto output_gate = at::zeros_like(old_cell);
  auto candidate_cell = at::zeros_like(old_cell);
  auto gates = at::cat({input_gate, output_gate, candidate_cell}, 1);

  const int threads = state_dim;
  const dim3 blocks(1, batch_size); // make sure state_dim < # threads

  AT_DISPATCH_FLOATING_TYPES(gates.type(), "lltm_forward_cuda", ([&] {
    lltm_fast_forward_kernel<scalar_t><<<blocks, threads>>>(
        X.data<scalar_t>(),
        weights.data<scalar_t>(),
        bias.data<scalar_t>(),
        gates.data<scalar_t>(),
        old_cell.data<scalar_t>(),
        new_h.data<scalar_t>(),
        new_cell.data<scalar_t>(),
        input_gate.data<scalar_t>(),
        output_gate.data<scalar_t>(),
        candidate_cell.data<scalar_t>(),
        input_dim + state_dim, //X_width
        state_dim);
  }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, X, gates};
}



// ===========================================
// Backward
// ===========================================


template <typename scalar_t>
__global__ void lltm_fast_backward_kernel(
    scalar_t* __restrict__ d_old_cell,
    scalar_t* __restrict__ d_gates,
    const scalar_t* __restrict__ grad_h,
    const scalar_t* __restrict__ grad_cell,
    const scalar_t* __restrict__ new_cell,
    const scalar_t* __restrict__ input_gate,
    const scalar_t* __restrict__ output_gate,
    const scalar_t* __restrict__ candidate_cell,
    const scalar_t* __restrict__ gate_weights,
    size_t state_size) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int index = blockIdx.y * state_size + column;
  const int gates_row = blockIdx.y * (state_size * 3);
  if (column < state_size) {
    const auto d_output_gate = tanh(new_cell[index]) * grad_h[index];
    const auto d_tanh_new_cell = output_gate[index] * grad_h[index];
    const auto d_new_cell =
        d_tanh(new_cell[index]) * d_tanh_new_cell + grad_cell[index];


    d_old_cell[index] = d_new_cell;
    const auto d_candidate_cell = input_gate[index] * d_new_cell;
    const auto d_input_gate = candidate_cell[index] * d_new_cell;


    const auto input_gate_index = gates_row + column;
    const auto output_gate_index = gates_row + state_size + column;
    const auto candidate_cell_index = gates_row + 2 * state_size + column;

    d_gates[input_gate_index] =
        d_input_gate * d_sigmoid(gate_weights[input_gate_index]);
    d_gates[output_gate_index] =
        d_output_gate * d_sigmoid(gate_weights[output_gate_index]);
    d_gates[candidate_cell_index] =
        d_candidate_cell * d_elu(gate_weights[candidate_cell_index]);
  }
}



std::vector<at::Tensor> lltm_fast_backward(
    at::Tensor grad_h,
    at::Tensor grad_cell,
    at::Tensor new_cell,
    at::Tensor input_gate,
    at::Tensor output_gate,
    at::Tensor candidate_cell,
    at::Tensor X,
    at::Tensor gate_weights,
    at::Tensor weights) {
  auto d_old_cell = at::zeros_like(new_cell);
  auto d_gates = at::zeros_like(gate_weights);

  const auto batch_size = new_cell.size(0);
  const auto state_size = new_cell.size(1);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(X.type(), "lltm_backward_cuda", ([&] {
    lltm_fast_backward_kernel<scalar_t><<<blocks, threads>>>(
        d_old_cell.data<scalar_t>(),
        d_gates.data<scalar_t>(),
        grad_h.contiguous().data<scalar_t>(),
        grad_cell.contiguous().data<scalar_t>(),
        new_cell.contiguous().data<scalar_t>(),
        input_gate.contiguous().data<scalar_t>(),
        output_gate.contiguous().data<scalar_t>(),
        candidate_cell.contiguous().data<scalar_t>(),
        gate_weights.contiguous().data<scalar_t>(),
        state_size);
  }));

  auto d_weights = d_gates.t().mm(X);
  auto d_bias = d_gates.sum(/*dim=*/0, /*keepdim=*/true);

  auto d_X = d_gates.mm(weights);
  auto d_old_h = d_X.slice(/*dim=*/1, 0, state_size);
  auto d_input = d_X.slice(/*dim=*/1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}


