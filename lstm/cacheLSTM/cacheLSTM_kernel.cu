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
__global__ void cache_lstm_forward_kernel(
    const scalar_t* __restrict__ XU, // (T, B, 4Dh)
    const scalar_t* __restrict__ W, // (4Dh, Dh)
    scalar_t* __restrict__ h, // (T+1, B, Dh)
    scalar_t* __restrict__ c0, // (B, Dh)
    size_t T,
    size_t B,
    size_t Dh) {

  const size_t batch_id = blockIdx.x;
  const size_t thread_id = threadIdx.x;

  // convenient constants
  const size_t B_Dh = B * Dh;
  const size_t B_Dh_4 = B_Dh * 4;
  const size_t b_Dh = batch_id * Dh;


  //extern __shared__ scalar_t shared_mem[];
  __shared__ scalar_t shared_mem[500]; // requires Dh <= 100

  scalar_t *ht = shared_mem; // (Dh)
  scalar_t *gates = &shared_mem[Dh]; // (4*Dh), saving activated gate values it, ft, ot, cbar

  // load weight vector used by current thread throughout time
  // *we set weight vector to be max 70 length, other than that we will exceed register cache capacity
  const int W_start = thread_id * Dh;
  scalar_t weights[70];
  memcpy(weights, &W[W_start], Dh * sizeof(scalar_t));

  // load initial cell value & h value (only thread_id < Dh does this)
  scalar_t prev_c;
  if (thread_id < Dh) {
    prev_c = c0[b_Dh + thread_id];
    ht[thread_id] = h[b_Dh + thread_id];
  }

  scalar_t xu; 
  size_t XU_start = b_Dh + thread_id;
  
  // start iterating through time
  for (int t = 0; t < T; t++) {

    // ----------------------------------------------------
    // step 1: load data into shared
    // get corresponding XU element
    xu = XU[XU_start];
    XU_start += B_Dh_4; 
    
    // sync
    __syncthreads();

    // -----------------------------------------------------
    // step 2: compute ht * weights + xu = gate
    scalar_t gate = xu;
    for (int i = 0; i<Dh; i++) {
      gate += ht[i] * weights[i];
    }

    // -----------------------------------------------------
    // step 3: activation and write to shared
    if (thread_id < 3 * Dh) {
      // use sigmoid
      gate = sigmoid(gate);
    } else {
      // use tanh
      gate = tanh(gate);
    }
    gates[thread_id] = gate;
    // sync
    __syncthreads();

    // -----------------------------------------------------
    // step 4: compute new cell and new h using 1/4 of the threads
    // hopefully 1/4 of the threads is still bigger than one warp, 
    // so other warps can rest and save resources for other blocks on same SM
    if (thread_id < Dh) { 
      // load 4 gates values
      // gate is ii
      scalar_t ff = gates[thread_id + Dh];
      scalar_t oo = gates[thread_id + Dh + Dh];
      scalar_t cbar = gates[thread_id + Dh + Dh + Dh];

      // compute new_c, new_h and write new_h to shared memory & global memory
      prev_c = sigmoid(ff * prev_c + gate * cbar);
      scalar_t new_h = tanh(prev_c) * oo;
      ht[thread_id] = new_h;
      h[(t+1) * B_Dh + b_Dh + thread_id] = new_h;
    }
    // sync
    __syncthreads();

  }

  // in the end, prev_c will be the cT value, write prev_c to c0 for output
  if (thread_id < Dh) {
    c0[b_Dh + thread_id] = prev_c; 
  }
}


// XU should be computed using pytorch: compute X * U which is time independent
// X: (T, B, Di)
// U: (Di, 4Dh)
// result: XU: (T, B, 4Dh)

std::vector<at::Tensor> cache_lstm_forward(
    at::Tensor XU, // (T, B, 4*Dh)
    at::Tensor W, // (4*Dh, Dh)
    at::Tensor h0, // (1, B, Dh), the initial hidden state c0
    at::Tensor c0  // (B, Dh), the initial cell state c0
){

  const auto T = XU.size(0); 
  const auto B = XU.size(1);
  const auto Dh = h0.size(2);

  // zero-pad h0 -> h
  auto hT = at::zeros({T, B, Dh}, h0.type());
  auto h = at::cat({h0, hT}, /*dim=*/0); // (T+1, B, Dh)

  const int blocks = B;
  const int threads = 4 * Dh;
  //const int shared_memory_per_block = (5 * Dh * sizeof(float));

  AT_DISPATCH_FLOATING_TYPES(h0.type(), "lstm_forward_cuda", ([&] {
    cache_lstm_forward_kernel<scalar_t><<<blocks, threads>>>(
        XU.data<scalar_t>(),
        W.data<scalar_t>(),
        h.data<scalar_t>(),
        c0.data<scalar_t>(),
        T, B, Dh
    );
  }));

  return {h, c0};
}



// ====================================================================================
// Backward
// ====================================================================================


template <typename scalar_t>
__global__ void cache_lstm_backward_kernel(
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



std::vector<at::Tensor> cache_lstm_backward(
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
    cache_lstm_backward_kernel<scalar_t><<<blocks, threads>>>(
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


