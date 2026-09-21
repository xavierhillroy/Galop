#ifndef EVAL_KERNELS_CUH
#define EVAL_KERNELS_CUH
#include <cstdint>

// =============================================================================
// Shared declarations for the two SR evaluation kernels. The DEFINITIONS live
// in EvaluatorCuda.cu (alongside the shared __device__ interpreter
// program_partial_sse and the __constant__ pool). They are declared here so the
// benchmark harness can launch the EXACT SAME kernels the backend uses -- which
// is what keeps the warp-vs-block comparison fair.
// =============================================================================

// prog_stride = words between consecutive programs in the flat buffer. Real use
// passes LGPConfig::MAX_PROGRAM_SIZE (the engine layout); the benchmark passes a
// tight stride (its fixed length) to pack more programs into memory.

// Warp-per-program: several warps packed per block (the high-occupancy design).
__global__ void evaluate_kernel(
    const uint32_t* __restrict__ instructions,
    const uint8_t*  __restrict__ lengths,
    float*          __restrict__ fitness_out,
    const float*    __restrict__ inputs,
    const float*    __restrict__ targets,
    int n_programs, int prog_stride, int N, int num_inputs);

// Block-per-program: one program per 32-thread block (the conventional baseline).
__global__ void evaluate_kernel_block(
    const uint32_t* __restrict__ instructions,
    const uint8_t*  __restrict__ lengths,
    float*          __restrict__ fitness_out,
    const float*    __restrict__ inputs,
    const float*    __restrict__ targets,
    int n_programs, int prog_stride, int N, int num_inputs);

// Packed (CSR) variants: programs concatenated with no padding; program i lives
// at instructions[cumL[i] : cumL[i+1]] and its length is the offset difference
// (no separate length array). cumL is int64 -- Sigma L overflows int32 at
// P ~ 1e8 x avg_len 40. Same interpreter, same reduction: fitness is expected
// bitwise-identical to the dense kernels for identical program content.
__global__ void evaluate_kernel_packed(
    const uint32_t*  __restrict__ instructions,
    const long long* __restrict__ cumL,
    float*           __restrict__ fitness_out,
    const float*     __restrict__ inputs,
    const float*     __restrict__ targets,
    int n_programs, int N, int num_inputs);

__global__ void evaluate_kernel_block_packed(
    const uint32_t*  __restrict__ instructions,
    const long long* __restrict__ cumL,
    float*           __restrict__ fitness_out,
    const float*     __restrict__ inputs,
    const float*     __restrict__ targets,
    int n_programs, int N, int num_inputs);

// Fill the __constant__ pool (d_constants) from LGPConfig::CONSTANTS. The
// backend constructor calls it; the benchmark calls it too, since it launches
// the kernels directly without constructing a CudaEvaluator.
void cuda_upload_constants();

#endif
