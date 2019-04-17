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
__global__ void lltm_fastseq_forward_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ old_h,
    scalar_t* __restrict__ old_cell,
    scalar_t* __restrict__ new_h,
    scalar_t* __restrict__ new_cell,
    scalar_t* __restrict__ input_gate,
    scalar_t* __restrict__ output_gate,
    scalar_t* __restrict__ candidate_cell,
    size_t input_len,
    size_t input_dim,
    size_t state_dim) {

  const size_t X_width = input_dim + state_dim;
  const size_t batch_id = blockIdx.y;
  const size_t column = threadIdx.x;
  const size_t index = batch_id * state_dim + column;
  //const size_t index = batch_id * state_dim + column;
  //const size_t gates_row = batch_id * (state_dim * 3);

  // input: (batch, input_len, state_dim + input_dim)
  // weights: (state_dim * 3, state_dim + input_dim)
  // bias: (state_dim * 3)

  if (column < state_dim) {
    // start iterate through time
    int i;
    int t;
    size_t old_h_start_idx = batch_id * state_dim;
    size_t x_start_idx = (batch_id * input_len - 1) * input_dim;

    size_t col_input = column;
    size_t col_output = state_dim + column;
    size_t col_cand = 2 * state_dim + column;

    size_t weights_row_start_hh_input = col_input * X_width;
    size_t weights_row_start_hh_output = col_output * X_width;
    size_t weights_row_start_hh_cand = col_cand * X_width;

    size_t weights_row_start_xx_input = weights_row_start_hh_input + state_dim;
    size_t weights_row_start_xx_output = weights_row_start_hh_output + state_dim;
    size_t weights_row_start_xx_cand = weights_row_start_hh_cand + state_dim;

    scalar_t input_gate_index_sum = 0;     
    scalar_t output_gate_index_sum = 0;     
    scalar_t cand_gate_index_sum = 0;

    auto bias_input = bias[col_input];
    auto bias_output = bias[col_output];
    auto bias_cand = bias[col_cand];

    for (t = 0; t < input_len; t++) {
      // ====================================================
      // prepare x_t of batch_id and at the current time step t
      // x_t = input[x_start_idx : x_start_idx + input_dim]
      x_start_idx += input_dim;
      auto *hh = &old_h[old_h_start_idx];
      auto *xx = &input[x_start_idx];
      auto *ww_hh_input = &weights[weights_row_start_hh_input];
      auto *ww_hh_output = &weights[weights_row_start_hh_output];
      auto *ww_hh_cand = &weights[weights_row_start_hh_cand];
      auto *ww_xx_input = &weights[weights_row_start_xx_input];
      auto *ww_xx_output = &weights[weights_row_start_xx_output];
      auto *ww_xx_cand = &weights[weights_row_start_xx_cand];
      
      // ==================================================================================
      // linear layer: [h x] * W + b
      //            -> input_gate, output_gate, candidate_cell
      
      // input_gate[batch, col] = sum_over_i(X[batch, i] * weights[col, i]) + bias[col]
      // output_gate[batch, col] = sum_over_i(X[batch, i] * weights[state_dim + col, i]) + bias[state_dim + col]
      // candidate_cell[batch, col] = sum_over_i(X[batch, i] * weights[2 * state_dim + col, i]) + bias[2 * state_dim + col]
      input_gate_index_sum = 0;     
      output_gate_index_sum = 0;     
      cand_gate_index_sum = 0;
      
      // dot product for 3 pairs of vectors
      // h_old x W
      for (i = 0; i < state_dim; i++) {
        auto hhh = *(hh++);
        input_gate_index_sum += hhh * (*(ww_hh_input++));
        output_gate_index_sum += hhh * (*(ww_hh_output++));
        cand_gate_index_sum += hhh * (*(ww_hh_cand++));
      }
      // x_t x W
      for (i = 0; i < input_dim; i++) {
        auto xxx = *(xx++);
        input_gate_index_sum += xxx * (*(ww_xx_input++));
        output_gate_index_sum += xxx * (*(ww_xx_output++));
        cand_gate_index_sum += xxx * (*(ww_xx_cand++));
      }
      // ===================================================================================
      // + b -> activation
      input_gate[index] = sigmoid(input_gate_index_sum + bias_input);
      output_gate[index] = sigmoid(output_gate_index_sum + bias_output);
      candidate_cell[index] = elu(cand_gate_index_sum + bias_cand); 

      // ===================================================================================
      // output layer
      new_cell[index] = old_cell[index] + candidate_cell[index] * input_gate[index];
      new_h[index] = tanh(new_cell[index]) * output_gate[index];

      // ----------------------------------------------------------------------------------
      // synchronize across all threads in this thread-block
      // then update h and cell
      __syncthreads();
      old_cell[index] = new_cell[index];
      old_h[index] = new_h[index];
      __syncthreads();

      // now we are ready to proceed to the next timestamp
    }
  }  
}


std::vector<at::Tensor> lltm_fastseq_forward(
    at::Tensor input,
    at::Tensor weights,
    at::Tensor bias,
    at::Tensor old_h,
    at::Tensor old_cell) {

  const auto batch_size = old_cell.size(0);
  const auto state_dim = old_cell.size(1);
  const auto input_dim = input.size(2); // input: (batch, input_len, input_dim)
  const auto input_len = input.size(1); 

  auto new_h = at::zeros_like(old_cell);
  auto new_cell = at::zeros_like(old_cell);
  auto input_gate = at::zeros_like(old_cell);
  auto output_gate = at::zeros_like(old_cell);
  auto candidate_cell = at::zeros_like(old_cell);

  const int threads = 512;
  const dim3 blocks(1, batch_size); // make sure state_dim < # threads

  AT_DISPATCH_FLOATING_TYPES(old_cell.type(), "lltm_forward_cuda", ([&] {
    lltm_fastseq_forward_kernel<scalar_t><<<blocks, threads>>>(
        input.data<scalar_t>(),
        weights.data<scalar_t>(),
        bias.data<scalar_t>(),
        old_h.data<scalar_t>(),
        old_cell.data<scalar_t>(),
        new_h.data<scalar_t>(),
        new_cell.data<scalar_t>(),
        input_gate.data<scalar_t>(),
        output_gate.data<scalar_t>(),
        candidate_cell.data<scalar_t>(),
        input_len,
        input_dim,
        state_dim);
  }));

  return {new_h, new_cell, input_gate, output_gate, candidate_cell, input};
}



// ====================================================================================
// Backward
// ====================================================================================


template <typename scalar_t>
__global__ void lltm_fastseq_backward_kernel(
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



std::vector<at::Tensor> lltm_fastseq_backward(
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
    lltm_fastseq_backward_kernel<scalar_t><<<blocks, threads>>>(
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


