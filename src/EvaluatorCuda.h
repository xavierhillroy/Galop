#ifndef EVALUATOR_CUDA_H
#define EVALUATOR_CUDA_H
#include "EvalBackend.h"
#include <vector>
#include <cstdlib>
#include <utility>
#include "Dataset.h"
#include <cstdint>

class CudaEvaluator : public EvalBackend {
public:
    // Parallelization strategy under test:
    //   Warp  = one program per WARP, several warps packed per block (high occupancy)
    //   Block = one program per BLOCK of 32 threads (the conventional baseline)
    enum class Strategy { Warp, Block };

    // Device memory layout for the instruction stream:
    //   Dense           = P x MAX_PROGRAM_SIZE padded slots (historical baseline)
    //   Packed          = CSR: programs concatenated, int64 cumL offsets, no padding
    //   PackedEffective = Packed, but only structurally-effective instructions
    //                     (introns removed at upload; bitwise-identical fitness)
    // Packing happens at the H2D boundary: the host engine keeps its dense
    // buffers, so evolution (and the golden) are untouched by construction.
    enum class Layout { Dense, Packed, PackedEffective };

    // warp_block_threads = threads/block for the Warp strategy (multiple of 32).
    // Ignored by Block (always 32). Defaults keep the engine's plain
    // construction (`make_unique<CudaEvaluator>()`) on the warp path.
    explicit CudaEvaluator(Strategy strategy = Strategy::Warp,
                           int warp_block_threads = 256,
                           Layout layout = Layout::Packed);

    // Cumulative wall-time spent inside evaluate_population (H2D + kernel +
    // D2H) -- everything else in a whole run is host-side evolution. Lets a
    // whole-run decompose into host vs eval without touching the CSV format.
    double eval_seconds() const { return eval_seconds_; }
    // Split of eval_seconds() (GALOP_PHASE_TIMERS set; timing only, no behavioural effect):
    // host packing of the population, (re)allocation of the packed buffers, host-to-device
    // upload, kernel from launch to synchronize, fitness download.
    double pack_seconds()   const { return pack_s_; }
    double alloc_seconds()  const { return alloc_s_; }
    double h2d_seconds()    const { return h2d_s_; }
    double kernel_seconds() const { return kernel_s_; }
    double d2h_seconds()    const { return d2h_s_; }

    // Per-call footprint log (one entry per evaluate_population call, i.e. per
    // generation): {Sigma L, Sigma L_eff} in instructions. L_eff is computed in
    // both packed modes (the analysis is cheap); Dense logs {Sigma L, 0}.
    const std::vector<std::pair<long long, long long>>& footprint_log() const {
        return footprint_log_;
    }
    ~CudaEvaluator() override;   // frees everything

    void evaluate_population(const uint32_t* instructions,
                             const uint8_t*  lengths,
                             float*          fitness_out,
                             int             n_programs,
                             const Dataset&  ds) override;

private:
    void upload_dataset(const Dataset& ds);   // runs once, behind the guard

    Strategy strategy_;
    int      warp_block_threads_;             // threads/block for the warp kernel
    Layout   layout_;

    // packed-path host staging (reused across generations) + device offsets
    // Host staging buffers are PINNED (cudaMallocHost): pageable transfers run
    // at roughly half PCIe bandwidth through a hidden staging copy; pinned
    // memory DMAs directly. Values are unchanged -- pure transfer speed.
    // Packed buffers are CONTENT-SIZED with high-water growth (2026-08-31):
    // capacity follows the actual packed volume Sigma L, so the device
    // footprint matches the paper's packing claim instead of the worst case
    // P*MAX_PROGRAM_SIZE. Allocation-only change: fitness bitwise-unchanged
    // (golden gate). ensure_packed_capacity() grows both staging + device.
    uint32_t*  h_packed_  = nullptr;
    long long  packed_capacity_ = 0;   // words currently allocated (host+device)
    long long* h_cumL_    = nullptr;
    float*     h_fitness_ = nullptr;   // pinned D2H staging for fitness
    long long* d_cumL_ = nullptr;             // [POPULATION_SIZE + 1]
    void ensure_packed_capacity(long long words);
    std::vector<std::pair<long long, long long>> footprint_log_;
    double eval_seconds_ = 0.0;
    const bool timers_ = std::getenv("GALOP_PHASE_TIMERS") != nullptr;
    double pack_s_ = 0.0, alloc_s_ = 0.0, h2d_s_ = 0.0, kernel_s_ = 0.0, d2h_s_ = 0.0;

    // ---- population buffers: allocated once, reused every generation ----
    uint32_t* d_instructions = nullptr;   // [TOTAL_INSTRUCTIONS]
    uint8_t*  d_lengths      = nullptr;   // [POPULATION_SIZE]
    float*    d_fitness      = nullptr;   // [POPULATION_SIZE]

    // ---- dataset: uploaded once, then resident for the whole run ----
    float*    d_inputs       = nullptr;   // [padded_N * num_inputs]
    float*    d_targets      = nullptr;   // [padded_N]
    int       ds_N           = 0;         // scalars travel as kernel args,
    int       ds_num_inputs  = 0;         // not device memory

    const Dataset* uploaded_ds_ = nullptr;  // the guard: "which dataset is resident"
};
#endif