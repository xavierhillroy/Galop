// =============================================================================
// cuda_diff.cu — differential test: GPU kernel vs CPU oracle.
//
// This is the GPU's equivalent of check_golden.sh. The golden (byte-exact)
// check does NOT apply to the GPU: float reductions reorder and host/device
// sinf differ by a few ULP, so the result matches only within a TOLERANCE.
// And the full evolve_sr TRAJECTORY diverges (chaos), so we never compare runs
// here -- we compare ONE evaluation of ONE population, both backends, and
// require max|Δ| < TOL.
//
// Over-TOL programs are not an automatic FAIL: evolved (bloated) programs can
// contain catastrophic-cancellation chains (e.g. repeated r2 -= r5 against a
// growing r5) that amplify few-ULP arithmetic differences -- host vs device
// sinf/cosf/div rounding -- into macroscopic fitness differences, and can
// then flip downstream comparisons (LT/GT). Such a program's float fitness is
// not a well-defined quantity at TOL precision. Every over-TOL program is
// therefore re-evaluated ON THE CPU ALONE in float AND double with the exact
// evaluator batching: if its own fitness moves > TOL under precision
// refinement, the program is precision-unstable and the CPU/GPU difference is
// attributed to that (WARN, not FAIL). A program that diverges from the
// oracle while being precision-STABLE indicates a real kernel defect -> FAIL.
// Mass divergence (>1% of the population) always FAILs regardless.
// (Mechanism established on evolved prog 679, seed 42: float-CPU 0.4546,
// double-CPU 0.4233, GPU-float 0.4698 -- three samples of the same chaos,
// while all bitwise layout identities held.)
//
// Run after every kernel change.
//
// Build:
//   nvcc -std=c++20 --expt-relaxed-constexpr -Isrc \
//        tests/cuda_diff.cu src/EvaluatorCuda.cu src/EvaluatorCpu.cpp \
//        src/Evaluator.cpp src/Interpreter.cpp src/Dataset.cpp src/LGPEngine.cpp \
//        -o cuda_diff
//   ./cuda_diff
// =============================================================================
#include "LGPEngine.h"
#include "ISA.h"
#include "EvaluatorCpu.h"
#include "EvaluatorCuda.h"
#include "Dataset.h"
#include "LGPConfig.h"
#include "cuda_errors.h"
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>

int main(int argc, char** argv) {
    Is_GPU_present();
    const char* ds_path = (argc > 1) ? argv[1] : "datasets/nguyen1_train.csv";

    // ---- a realistic random population (every opcode exercised) ------------
    // init_evolution() fills the live buffer with random valid programs --
    // the exact distribution real runs start from. (All length 8 here; to
    // stress varied lengths later, evolve a few generations first.)
    LGPEngine engine(42);
    engine.init_evolution();

    const auto& data    = engine.get_data();
    const int   buf     = engine.current_buffer_index();
    const uint32_t* instr   = data.instructions_buf[buf].data();
    const uint8_t*  lengths = data.program_lengths_buf[buf].data();

    // ---- a real dataset (ss_tot computed, so R² is meaningful) -------------
    Dataset ds = load_csv_1d(ds_path);
    const int n = LGPConfig::POPULATION_SIZE;
    printf("Population: %d programs | dataset N=%d num_inputs=%d\n",
           n, ds.N, ds.num_inputs);

    // ---- CPU oracle: the reference fitness ---------------------------------
    std::vector<float> fit_cpu(n);
    CpuEvaluator{}.evaluate_population(instr, lengths, fit_cpu.data(), n, ds);

    // How many programs actually have positive fitness? If ~0, the diff is
    // comparing 0==0 everywhere and proves nothing -- need an evolved population.
    int nz = 0; float mx = 0.0f;
    for (int i = 0; i < n; ++i) { if (fit_cpu[i] > 0.0f) ++nz; mx = std::max(mx, fit_cpu[i]); }
    printf("CPU fitness: %d/%d programs > 0, max = %.6f\n", nz, n, mx);

    // ---- check one GPU strategy x layout against the oracle ----------------
    const float TOL = 1e-4f;
    auto run_gpu = [&](CudaEvaluator::Strategy strat, CudaEvaluator::Layout lay,
                       std::vector<float>& out) {
        CudaEvaluator gpu(strat, 256, lay);
        gpu.evaluate_population(instr, lengths, out.data(), n, ds);
    };
    // CPU-only precision-stability probe: evaluate program i with float and
    // with double registers (same ops, same evaluator batching). Returns the
    // float and double fitness; a gap > TOL means the program's float fitness
    // is rounding-chaotic, so host/device ULP differences legitimately explode.
    auto dual_precision_fitness = [&](int i, float& fit_f, float& fit_d) {
        const uint32_t* p = instr + (size_t)i * LGPConfig::MAX_PROGRAM_SIZE;
        const int L = (int)lengths[i];
        constexpr int CH = LGPConfig::NUM_CONTEXTS;
        constexpr int NR = LGPConfig::NUM_REGISTERS;
        double sse_f = 0.0, sse_d = 0.0;
        for (int base = 0; base < ds.N; base += CH) {
            const int valid = std::min(CH, ds.N - base);
            float  fr[CH * NR];
            double dr[CH * NR];
            for (int c = 0; c < CH; ++c) {
                for (int k = 0; k < ds.num_inputs; ++k) {
                    fr[c * NR + k] = ds.inputs[(base + c) * ds.num_inputs + k];
                    dr[c * NR + k] = (double)fr[c * NR + k];
                }
                for (int k = ds.num_inputs; k < NR; ++k) { fr[c * NR + k] = 0.0f; dr[c * NR + k] = 0.0; }
            }
            for (int pc = 0; pc < L; ++pc) {
                const uint32_t w = p[pc];
                const int op = ISA::get_op(w), d = ISA::get_dest_index(w),
                          s1 = ISA::get_src1_index(w), s2 = ISA::get_src2_index(w);
                const bool k2 = ISA::is_src2_constant(w);
                for (int c = 0; c < CH; ++c) {
                    const float  fa = fr[c * NR + s1];
                    const float  fb = k2 ? LGPConfig::CONSTANTS[s2] : fr[c * NR + s2];
                    const double da = dr[c * NR + s1];
                    const double db = k2 ? (double)LGPConfig::CONSTANTS[s2] : dr[c * NR + s2];
                    fr[c * NR + d] = ISA::apply_op((uint8_t)op, fa, fb);
                    double dv;
                    switch (op) {
                        case ISA::ADD: dv = da + db; break;
                        case ISA::SUB: dv = da - db; break;
                        case ISA::MUL: dv = da * db; break;
                        case ISA::DIV: dv = da / db; break;
                        case ISA::SIN: dv = sin(da); break;
                        case ISA::COS: dv = cos(da); break;
                        case ISA::LT:  dv = (da < db) ? 1.0 : 0.0; break;
                        default:       dv = (da > db) ? 1.0 : 0.0; break;
                    }
                    dr[c * NR + d] = dv;
                }
            }
            for (int c = 0; c < valid; ++c) {
                const double t  = (double)ds.targets[base + c];
                const double ef = (double)fr[c * NR] - t;
                const double ed = dr[c * NR] - t;
                sse_f += ef * ef; sse_d += ed * ed;
            }
        }
        const float mse_f = (float)(sse_f / ds.N), mse_d = (float)(sse_d / ds.N);
        fit_f = std::isfinite(mse_f) ? 1.0f / (1.0f + mse_f) : 0.0f;
        fit_d = std::isfinite(mse_d) ? 1.0f / (1.0f + mse_d) : 0.0f;
    };
    auto check = [&](const char* name, CudaEvaluator::Strategy strat,
                     CudaEvaluator::Layout lay, std::vector<float>& keep) -> bool {
        run_gpu(strat, lay, keep);
        float max_diff = 0.0f;
        int   worst = -1;
        std::vector<int> over;
        for (int i = 0; i < n; ++i) {
            const float d = fabsf(fit_cpu[i] - keep[i]);
            if (d > max_diff) { max_diff = d; worst = i; }
            if (d > TOL) over.push_back(i);
        }
        const int wi = worst < 0 ? 0 : worst;   // guard: worst stays -1 iff all diffs are 0
        if (over.empty()) {
            printf("[%-14s] max |Δ| = %.3e at prog %d (cpu=%.7f gpu=%.7f)  0/%d over TOL  -> PASS\n",
                   name, max_diff, worst, fit_cpu[wi], keep[wi], n);
            return true;
        }
        // classify every over-TOL program
        int chaotic = 0;
        for (int i : over) {
            float ff, fd;
            dual_precision_fitness(i, ff, fd);
            const bool is_chaotic = fabsf(fd - ff) > TOL;
            chaotic += is_chaotic;
            printf("[%-14s]   prog %d over TOL: |Δ|=%.3e  CPU float=%.7f double=%.7f "
                   "(precision jump %.3e)  -> %s\n",
                   name, i, fabsf(fit_cpu[i] - keep[i]), ff, fd, fabsf(fd - ff),
                   is_chaotic ? "precision-unstable (attributed)"
                              : "PRECISION-STABLE (kernel defect!)");
        }
        const bool pass = chaotic == (int)over.size() && (int)over.size() <= n / 100;
        printf("[%-14s] max |Δ| = %.3e at prog %d (cpu=%.7f gpu=%.7f)  %d/%d over TOL "
               "(%d chaos-attributed)  -> %s\n",
               name, max_diff, worst, fit_cpu[wi], keep[wi],
               (int)over.size(), n, chaotic,
               pass ? "PASS (see WARN lines)" : "FAIL");
        return pass;
    };

    std::vector<float> f_wd(n), f_wp(n), f_we(n), f_bd(n), f_bp(n);
    bool ok = check("warp/dense",   CudaEvaluator::Strategy::Warp,  CudaEvaluator::Layout::Dense,           f_wd);
    ok     &= check("warp/packed",  CudaEvaluator::Strategy::Warp,  CudaEvaluator::Layout::Packed,          f_wp);
    ok     &= check("warp/effect",  CudaEvaluator::Strategy::Warp,  CudaEvaluator::Layout::PackedEffective, f_we);
    ok     &= check("block/dense",  CudaEvaluator::Strategy::Block, CudaEvaluator::Layout::Dense,           f_bd);
    ok     &= check("block/packed", CudaEvaluator::Strategy::Block, CudaEvaluator::Layout::Packed,          f_bp);

    // ---- BITWISE identity requirements --------------------------------------
    // dense vs packed: identical program content, only address arithmetic differs.
    // packed vs packed-effective: intron removal is exact on straight-line code.
    auto bitwise = [&](const char* name, const std::vector<float>& a,
                       const std::vector<float>& b) -> bool {
        const bool same = std::memcmp(a.data(), b.data(), n * sizeof(float)) == 0;
        printf("[%-14s] %s\n", name, same ? "BITWISE IDENTICAL -> PASS"
                                          : "DIFFERS -> FAIL");
        return same;
    };
    ok &= bitwise("dense==packed",  f_wd, f_wp);
    ok &= bitwise("packed==effect", f_wp, f_we);
    ok &= bitwise("blk dns==pckd",  f_bd, f_bp);

    if (std::getenv("GALOP_DIFF_QUICK")) {
        printf(ok ? "QUICK PASS (phase 1 only)\n" : "FAIL\n");
        return ok ? 0 : 1;
    }

    // ---- phase 2: EVOLVED population (varied lengths ~ up to 50, heavy
    // introns) -- the real stress test for packed offsets + intron removal ----
    printf("\n-- evolving %d generations on CPU for a bloated population --\n",
           LGPConfig::MAX_GENERATIONS);
    engine.evolve_sr(ds);
    const int   buf2     = engine.current_buffer_index();
    instr   = data.instructions_buf[buf2].data();
    lengths = data.program_lengths_buf[buf2].data();
    long long sumL = 0;
    for (int i = 0; i < n; ++i) sumL += lengths[i];
    printf("evolved: mean length %.1f\n", (double)sumL / n);

    CpuEvaluator{}.evaluate_population(instr, lengths, fit_cpu.data(), n, ds);
    ok &= check("E warp/dense",  CudaEvaluator::Strategy::Warp, CudaEvaluator::Layout::Dense,           f_wd);
    ok &= check("E warp/packed", CudaEvaluator::Strategy::Warp, CudaEvaluator::Layout::Packed,          f_wp);
    ok &= check("E warp/effect", CudaEvaluator::Strategy::Warp, CudaEvaluator::Layout::PackedEffective, f_we);
    ok &= bitwise("E dense==packed",  f_wd, f_wp);
    ok &= bitwise("E packed==effect", f_wp, f_we);
    {   // report the footprint the effective layout actually achieved
        CudaEvaluator gpu(CudaEvaluator::Strategy::Warp, 256, CudaEvaluator::Layout::PackedEffective);
        std::vector<float> tmp(n);
        gpu.evaluate_population(instr, lengths, tmp.data(), n, ds);
        auto [L, Leff] = gpu.footprint_log().back();
        printf("evolved footprint: Sigma L=%lld, Sigma L_eff=%lld (effective fraction %.1f%%)\n",
               L, Leff, 100.0 * Leff / L);
    }

    printf(ok ? "ALL PASS: every kernel/layout agrees with the oracle; "
                "packed & intron-removed are bitwise identical (init + evolved).\n"
              : "FAIL: divergence detected.\n");
    return ok ? 0 : 1;
}
