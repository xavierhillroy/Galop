#include "EvaluatorCuda.h"
#include "eval_kernels.cuh"
#include "LGPConfig.h"
#include "cuda_errors.h"
#include "ISA.h"
#include "Introns.h"
#include <cstring>
#include <cassert>
#include <chrono>

__constant__ float d_constants[LGPConfig::NUM_CONSTANTS];

// Fill the constant pool from the host config. Shared by the backend ctor and
// the benchmark (which launches the kernels without constructing a backend).
void cuda_upload_constants() {
    ERR(cudaMemcpyToSymbol(d_constants, LGPConfig::CONSTANTS.data(),
                           LGPConfig::NUM_CONSTANTS * sizeof(float)));
}

// ---- allocate ONCE (full size, so it serves gen-0 full + later non-elite) --
CudaEvaluator::CudaEvaluator(Strategy strategy, int warp_block_threads, Layout layout)
    : strategy_(strategy), warp_block_threads_(warp_block_threads), layout_(layout)
{
    if (layout_ != Layout::Dense) {
        // Packed instruction buffers (host pinned + device): PRE-ALLOCATION
        // FLOOR policy. When the worst case P*MAX_PROGRAM_SIZE is small
        // (<= 2 GB — every timing-grid cell), pre-allocate it here, outside
        // any timed region, exactly as the pre-fix engine did: timing
        // behaviour is identical by construction (A/B verified). Beyond the
        // floor (capacity-scale P), buffers are content-sized: grown to the
        // generation's actual Sigma L by ensure_packed_capacity(), so the
        // device footprint scales with packed content, not the worst case.
        // Floor covers every RECORDED timing/scale cell (grid P<=1e6, scale
        // rows 2^24 and 1e8): their allocation stays in the ctor exactly as
        // when they were measured (A/B: deltas 0.0-0.3%, within noise).
        // Capacity-ladder rungs (P>=2e8) exceed the floor and get
        // content-sized buffers — the packing-footprint claim at scale.
        constexpr long long PREALLOC_FLOOR_WORDS = 6LL * 1000 * 1000 * 1000;  // 24 GB ~ P=1.2e8
        if (LGPConfig::TOTAL_INSTRUCTIONS <= PREALLOC_FLOOR_WORDS) {
            ERR(cudaMallocHost(&h_packed_,
                (size_t)LGPConfig::TOTAL_INSTRUCTIONS * sizeof(uint32_t)));
            ERR(cudaMalloc(&d_instructions,
                (size_t)LGPConfig::TOTAL_INSTRUCTIONS * sizeof(uint32_t)));
            packed_capacity_ = LGPConfig::TOTAL_INSTRUCTIONS;
        }
        ERR(cudaMallocHost(&h_cumL_,
                           (size_t)(LGPConfig::POPULATION_SIZE + 1) * sizeof(long long)));
        ERR(cudaMalloc(&d_cumL_,
                       (LGPConfig::POPULATION_SIZE + 1) * sizeof(long long)));
    }
    ERR(cudaMallocHost(&h_fitness_, (size_t)LGPConfig::POPULATION_SIZE * sizeof(float)));
    if (layout_ == Layout::Dense) {
        ERR(cudaMalloc(&d_instructions,
                       (size_t)LGPConfig::TOTAL_INSTRUCTIONS * sizeof(uint32_t)));
    }
    ERR(cudaMalloc(&d_lengths,
                          LGPConfig::POPULATION_SIZE * sizeof(uint8_t)));
    ERR(cudaMalloc(&d_fitness,
                          LGPConfig::POPULATION_SIZE * sizeof(float)));

    cuda_upload_constants();   // fill the __constant__ pool (shared helper)
}

// Grow the packed staging (pinned host) + device instruction buffers to hold
// `words` instruction words. High-water: never shrinks; 25% headroom so early
// growth toward the length cap does not realloc every generation. Allocation
// only — contents are fully rewritten by the pack each generation, so results
// are independent of when (or whether) a regrow happens.
void CudaEvaluator::ensure_packed_capacity(long long words) {
    if (words <= packed_capacity_) return;
    const auto t_a0 = std::chrono::high_resolution_clock::now();
    long long cap = words + words / 4;
    if (cap > LGPConfig::TOTAL_INSTRUCTIONS) cap = LGPConfig::TOTAL_INSTRUCTIONS;
    if (cap < words) cap = words;
    if (h_packed_)       cudaFreeHost(h_packed_);
    if (d_instructions)  cudaFree(d_instructions);
    ERR(cudaMallocHost(&h_packed_, (size_t)cap * sizeof(uint32_t)));
    ERR(cudaMalloc(&d_instructions, (size_t)cap * sizeof(uint32_t)));
    packed_capacity_ = cap;
    if (timers_) alloc_s_ += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_a0).count();
}

CudaEvaluator::~CudaEvaluator() {
    cudaFreeHost(h_packed_);
    cudaFreeHost(h_cumL_);
    cudaFreeHost(h_fitness_);
    cudaFree(d_cumL_);
    cudaFree(d_instructions);
    cudaFree(d_lengths);
    cudaFree(d_fitness);
    cudaFree(d_inputs);
    cudaFree(d_targets);
}

// ---- dataset upload: called once, guarded by object identity --------------
void CudaEvaluator::upload_dataset(const Dataset& ds) {
    const int padded = ds.padded_N();
    // NOTE: case-major upload is coalesced only because num_inputs==1.
    // For multi-input datasets, transpose to input-major (SoA) here so each
    // per-dimension warp load stays coalesced. See GPU_PORT_PLAN. ~ basically would create one array for each inpuy 
    ERR(cudaMalloc(&d_inputs,  padded * ds.num_inputs * sizeof(float)));
    ERR(cudaMalloc(&d_targets, padded * sizeof(float)));
    ERR(cudaMemcpy(d_inputs,  ds.inputs.data(),
                          padded * ds.num_inputs * sizeof(float),
                          cudaMemcpyHostToDevice));
    ERR(cudaMemcpy(d_targets, ds.targets.data(),
                          padded * sizeof(float), cudaMemcpyHostToDevice));
    ds_N          = ds.N;
    ds_num_inputs = ds.num_inputs;
}



// INTERPRETER FOR WARP
__device__ float program_partial_sse(
        const uint32_t* prog, int len, int lane,
        const float* __restrict__ inputs, const float* __restrict__ targets,
        int N, int num_inputs)
{
    float sse_lane = 0.0f;                      // float: GP selection is robust to
                                                // last-digit fitness noise, and FP32
                                                // is far faster (esp. consumer FP64)
    for (int base = 0; base < N; base += 32) { // multiples of 32 warps (warp def not best appraoch when SR > 32 LOL  )
        const int caseIdx = base + lane; // current lane 
        if (caseIdx >= N) continue;            // replaces the CPU's `valid` clamp

        float reg[LGPConfig::NUM_REGISTERS];
        assert(num_inputs<= LGPConfig::NUM_REGISTERS);
        // -------------------------------------------------------------
        // store inputs 
        for (int inp = 0; inp <num_inputs; inp ++){
            reg[inp] = inputs[caseIdx * num_inputs + inp];
        }
        // zero out the rest of regs
        for (int inp = num_inputs; inp < LGPConfig::NUM_REGISTERS; inp ++){
            reg[inp] = 0.0f;
        }
        // executing instruction

        // loop through instructions 
        for (int pc = 0; pc < len; ++pc){
            // decdode instruction
            uint32_t cur_instruction = prog[pc];
            uint8_t dest_index = ISA::get_dest_index(cur_instruction);
            uint8_t op = ISA::get_op(cur_instruction);
            uint8_t src1_index = ISA::get_src1_index(cur_instruction);
            uint8_t src2_index = ISA::get_src2_index(cur_instruction);
            bool src2_const = ISA::is_src2_constant(cur_instruction);

            // apply
            float a = reg[src1_index];
            float b = src2_const ? d_constants[src2_index] : reg[src2_index];
            reg[dest_index] = ISA::apply_op(op,a,b);
            
        }// loops through instructions 

        const float err = reg[LGPConfig::OUTPUT_REG] - targets[caseIdx];
        sse_lane += err * err;
    }
    return sse_lane;
}
// big daddy population kernel 
__global__ void evaluate_kernel(
        const uint32_t* __restrict__ instructions,
        const uint8_t*  __restrict__ lengths,
        float*          __restrict__ fitness_out,
        const float*    __restrict__ inputs,
        const float*    __restrict__ targets,
        int n_programs, int prog_stride, int N, int num_inputs)
{
    const int warps_per_block = blockDim.x >> 5; // block dim ID /32
    const int lane = threadIdx.x & 31; // Threadidx %32
    const int prog = blockIdx.x * warps_per_block + (threadIdx.x >> 5); // which blocks came before * number of waprs in the block + threadID/ 32- this decides what warp I am in the instruction thing
    if (prog >= n_programs) return;            // warp-uniform: all 32 lanes share prog - if more warps then progs

    float sse = program_partial_sse(
        instructions + (size_t)prog * prog_stride,
        lengths[prog], lane, inputs, targets, N, num_inputs);// device func the interpreter path

    // ---- warp reduction: 32 partials -> lane 0 ----
    for (int off = 16; off > 0; off >>= 1)
        sse += __shfl_down_sync(0xffffffffu, sse, off); // this is the tree reduction

    if (lane == 0) {                           // must replicate evaluate_sr_r2 + r2_to_fitness - produces refined fitness not just raw perfectly comparable
        float fit;
        if (!isfinite(sse)) fit = 0.0f;
        else fit = 1.0f / (1.0f + sse / (float)N);   // MSE -> fitness (mse_to_fitness)
        fitness_out[prog] = fit;
    }
}

// ---------------------------------------------------------------------------
// BLOCK baseline: one program per BLOCK of 32 threads. Same interpreter
// (program_partial_sse), but the scheduling unit is a BLOCK -- so this is
// limited by the SM's resident-BLOCK ceiling, not its warp ceiling. That is
// exactly the comparison: at large populations the warp kernel sustains higher
// occupancy than this. Reduction uses shared memory + __syncthreads (the
// conventional block-cooperative pattern) rather than __shfl.
// ---------------------------------------------------------------------------
__global__ void evaluate_kernel_block(
        const uint32_t* __restrict__ instructions,
        const uint8_t*  __restrict__ lengths,
        float*          __restrict__ fitness_out,
        const float*    __restrict__ inputs,
        const float*    __restrict__ targets,
        int n_programs, int prog_stride, int N, int num_inputs)
{
    const int prog = blockIdx.x;     // one program per block
    const int lane = threadIdx.x;    // 0..31 (block is launched with 32 threads)
    if (prog >= n_programs) return;  // block-uniform: all 32 threads share blockIdx

    float sse = program_partial_sse(
        instructions + (size_t)prog * prog_stride,
        lengths[prog], lane, inputs, targets, N, num_inputs);

    // ---- shared-memory tree reduction across the 32 threads -> thread 0 ----
    // Same pairing order as the warp __shfl reduction, so results are identical.
    __shared__ float sdata[32];
    sdata[lane] = sse;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) {
        if (lane < s) sdata[lane] += sdata[lane + s];
        __syncthreads();
    }

    if (lane == 0) {                 // identical finalization to the warp kernel
        const float total = sdata[0];
        float fit;
        if (!isfinite(total)) fit = 0.0f;
        else fit = 1.0f / (1.0f + total / (float)N);   // MSE -> fitness (mse_to_fitness)
        fitness_out[prog] = fit;
    }
}

// ---------------------------------------------------------------------------
// PACKED (CSR) kernels: identical interpreter + reduction, but program i is
// read from instructions[cumL[i] : cumL[i+1]] instead of a padded stride.
// Only the address arithmetic differs, so fitness is bitwise-identical to the
// dense kernels for the same program content.
// ---------------------------------------------------------------------------
__global__ void evaluate_kernel_packed(
        const uint32_t*  __restrict__ instructions,
        const long long* __restrict__ cumL,
        float*           __restrict__ fitness_out,
        const float*     __restrict__ inputs,
        const float*     __restrict__ targets,
        int n_programs, int N, int num_inputs)
{
    const int warps_per_block = blockDim.x >> 5;
    const int lane = threadIdx.x & 31;
    const int prog = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    if (prog >= n_programs) return;

    const long long lo = cumL[prog], hi = cumL[prog + 1];
    float sse = program_partial_sse(instructions + lo, (int)(hi - lo),
                                    lane, inputs, targets, N, num_inputs);
    for (int off = 16; off > 0; off >>= 1)
        sse += __shfl_down_sync(0xffffffffu, sse, off);
    if (lane == 0) {
        float fit;
        if (!isfinite(sse)) fit = 0.0f;
        else fit = 1.0f / (1.0f + sse / (float)N);
        fitness_out[prog] = fit;
    }
}

__global__ void evaluate_kernel_block_packed(
        const uint32_t*  __restrict__ instructions,
        const long long* __restrict__ cumL,
        float*           __restrict__ fitness_out,
        const float*     __restrict__ inputs,
        const float*     __restrict__ targets,
        int n_programs, int N, int num_inputs)
{
    const int prog = blockIdx.x;
    const int lane = threadIdx.x;
    if (prog >= n_programs) return;

    const long long lo = cumL[prog], hi = cumL[prog + 1];
    float sse = program_partial_sse(instructions + lo, (int)(hi - lo),
                                    lane, inputs, targets, N, num_inputs);
    __shared__ float sdata[32];
    sdata[lane] = sse;
    __syncthreads();
    for (int s = 16; s > 0; s >>= 1) {
        if (lane < s) sdata[lane] += sdata[lane + s];
        __syncthreads();
    }
    if (lane == 0) {
        const float total = sdata[0];
        float fit;
        if (!isfinite(total)) fit = 0.0f;
        else fit = 1.0f / (1.0f + total / (float)N);
        fitness_out[prog] = fit;
    }
}

void CudaEvaluator::evaluate_population(const uint32_t* instructions,
                                        const uint8_t*  lengths,
                                        float*          fitness_out,
                                        int             n_programs,
                                        const Dataset&  ds)
{
    const auto t_eval0 = std::chrono::high_resolution_clock::now();
    using _clk = std::chrono::high_resolution_clock;
    _clk::time_point tA{}, tU0{}, tU1{}, tK0{}, tK1{}, tD0{};
    const double alloc_before = alloc_s_;
    // 1) dataset — upload the first time we see it, then it stays resident
    if (uploaded_ds_ != &ds) {
        upload_dataset(ds);
        uploaded_ds_ = &ds;
    }
    if (timers_) tA = _clk::now();

    // 2) population UP. The caller (evaluate_range) already offset these
    //    pointers past the elites, so we copy n_programs worth from the FRONT
    //    of the host pointers into the FRONT of the device buffers. The device
    //    works in a 0-based window; the host offset handles placement.
    //
    //    Dense: copy the padded slots as-is. Packed: scan+scatter at this H2D
    //    boundary (host engine stays dense, so evolution/golden are untouched);
    //    PackedEffective additionally strips structural introns -- bitwise-
    //    identical fitness, smaller Sigma L on device.
    if (layout_ == Layout::Dense) {
        if (timers_) tU0 = _clk::now();
        ERR(cudaMemcpy(d_instructions, instructions,
                       (size_t)n_programs * LGPConfig::MAX_PROGRAM_SIZE * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
        ERR(cudaMemcpy(d_lengths, lengths, n_programs * sizeof(uint8_t),
                       cudaMemcpyHostToDevice));
        if (timers_) tU1 = _clk::now();
        long long sumL = 0;
        for (int i = 0; i < n_programs; ++i) sumL += lengths[i];
        footprint_log_.push_back({sumL, 0});
    } else if (layout_ == Layout::Packed) {
        // Two-phase parallel pack: chunked length prefix (fixed chunk count ->
        // offsets independent of thread count), then parallel scatter. Output
        // is byte-identical to the historical serial loop.
        constexpr int C = 256;
        long long chunk_sum[C], chunk_off[C];
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < C; ++c) {
            const long long lo = (long long)n_programs * c / C,
                            hi = (long long)n_programs * (c + 1) / C;
            long long s = 0;
            for (long long i = lo; i < hi; ++i) s += lengths[i];
            chunk_sum[c] = s;
        }
        long long off = 0;
        for (int c = 0; c < C; ++c) { chunk_off[c] = off; off += chunk_sum[c]; }
        const long long sumL = off;
        ensure_packed_capacity(sumL);
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < C; ++c) {
            const long long lo = (long long)n_programs * c / C,
                            hi = (long long)n_programs * (c + 1) / C;
            long long o = chunk_off[c];
            for (long long i = lo; i < hi; ++i) {
                const uint32_t* src = instructions + (size_t)i * LGPConfig::MAX_PROGRAM_SIZE;
                const int L = lengths[i];
                h_cumL_[i] = o;
                std::memcpy(h_packed_ + o, src, (size_t)L * sizeof(uint32_t));
                o += L;
            }
        }
        h_cumL_[n_programs] = off;
        footprint_log_.push_back({sumL, sumL});
        if (timers_) tU0 = _clk::now();
        ERR(cudaMemcpy(d_instructions, h_packed_, (size_t)off * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
        ERR(cudaMemcpy(d_cumL_, h_cumL_, (size_t)(n_programs + 1) * sizeof(long long),
                       cudaMemcpyHostToDevice));
        if (timers_) tU1 = _clk::now();
    } else {  // PackedEffective: two-phase like Packed. Phase A walks the
        // liveness masks only, so the buffers are sized by the true effective
        // total -- sizing by the raw bound would give this layout the same
        // allocation footprint as Packed and collapse its capacity ceiling
        // onto Packed's. Chunked prefix offsets: output byte-identical to the
        // historical serial loop for any thread count.
        constexpr int C = 256;
        long long chunk_eff[C], chunk_raw[C], chunk_off[C];
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < C; ++c) {
            const long long lo = (long long)n_programs * c / C,
                            hi = (long long)n_programs * (c + 1) / C;
            long long e = 0, s = 0;
            char mask[LGPConfig::MAX_PROGRAM_SIZE];
            for (long long i = lo; i < hi; ++i) {
                const int L = lengths[i];
                s += L;
                e += Introns::effective_mask(
                    instructions + (size_t)i * LGPConfig::MAX_PROGRAM_SIZE, L, mask);
            }
            chunk_eff[c] = e; chunk_raw[c] = s;
        }
        long long off = 0, sumL = 0;
        for (int c = 0; c < C; ++c) {
            chunk_off[c] = off; off += chunk_eff[c]; sumL += chunk_raw[c];
        }
        const long long effL = off;
        ensure_packed_capacity(effL);
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < C; ++c) {
            const long long lo = (long long)n_programs * c / C,
                            hi = (long long)n_programs * (c + 1) / C;
            long long o = chunk_off[c];
            for (long long i = lo; i < hi; ++i) {
                h_cumL_[i] = o;
                o += Introns::extract_effective(
                    instructions + (size_t)i * LGPConfig::MAX_PROGRAM_SIZE,
                    lengths[i], h_packed_ + o);
            }
        }
        h_cumL_[n_programs] = effL;
        footprint_log_.push_back({sumL, effL});
        if (timers_) tU0 = _clk::now();
        ERR(cudaMemcpy(d_instructions, h_packed_, (size_t)off * sizeof(uint32_t),
                       cudaMemcpyHostToDevice));
        ERR(cudaMemcpy(d_cumL_, h_cumL_, (size_t)(n_programs + 1) * sizeof(long long),
                       cudaMemcpyHostToDevice));
        if (timers_) tU1 = _clk::now();
    }

    // 3) launch — pick the kernel under test (strategy x layout)
    if (timers_) tK0 = _clk::now();
    if (strategy_ == Strategy::Warp) {
        const int threads = warp_block_threads_;             // multiple of 32
        const int warps_per_block = threads / 32;
        // ceil in 64-bit: at P near INT_MAX the int addition overflows before
        // the divide (P + 7 wraps negative), which would misreport the true
        // 2^31-1 index bound as a launch crash
        const int blocks = (int)(((long long)n_programs + warps_per_block - 1) / warps_per_block);
        if (layout_ == Layout::Dense)
            evaluate_kernel<<<blocks, threads>>>(
                d_instructions, d_lengths, d_fitness, d_inputs, d_targets,
                n_programs, LGPConfig::MAX_PROGRAM_SIZE, ds_N, ds_num_inputs);
        else
            evaluate_kernel_packed<<<blocks, threads>>>(
                d_instructions, d_cumL_, d_fitness, d_inputs, d_targets,
                n_programs, ds_N, ds_num_inputs);
    } else { // Strategy::Block — one program per block of 32 threads
        if (layout_ == Layout::Dense)
            evaluate_kernel_block<<<n_programs, 32>>>(
                d_instructions, d_lengths, d_fitness, d_inputs, d_targets,
                n_programs, LGPConfig::MAX_PROGRAM_SIZE, ds_N, ds_num_inputs);
        else
            evaluate_kernel_block_packed<<<n_programs, 32>>>(
                d_instructions, d_cumL_, d_fitness, d_inputs, d_targets,
                n_programs, ds_N, ds_num_inputs);
    }
    ERR(cudaGetLastError());        // catch launch-config errors
    ERR(cudaDeviceSynchronize());   // wait + catch runtime errors
    if (timers_) { tK1 = _clk::now(); tD0 = tK1; }

    // 4) fitness DOWN — pinned staging, then host-side copy to the caller's
    //    (pageable) buffer: two fast copies beat one pageable-D2H.
    ERR(cudaMemcpy(h_fitness_, d_fitness,
                          n_programs * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::memcpy(fitness_out, h_fitness_, (size_t)n_programs * sizeof(float));
    const auto t_end = std::chrono::high_resolution_clock::now();
    eval_seconds_ += std::chrono::duration<double>(t_end - t_eval0).count();
    if (timers_) {
        auto sec = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        pack_s_   += sec(tA, tU0) - (alloc_s_ - alloc_before);   // host packing of this upload, minus any (re)allocation inside it
        h2d_s_    += sec(tU0, tU1);
        kernel_s_ += sec(tK0, tK1);                               // launch to synchronize
        d2h_s_    += sec(tD0, t_end);
    }
}