// =============================================================================
// cuda_evo_bench.cu — WHOLE-RUN throughput over a full evolutionary run.
//
// Host-side evolutionary loop (selection/variation on CPU), evaluation on GPU.
// Reports Langdon/EvoGP GPops/s (RAW operations per second):
//     GPops/s = ( sum_g  P · S_bar_g · D ) / T
//
// POPULATION_SIZE and MAX_GENERATIONS come from LGPConfig (override at compile
// time with -DPOPULATION_SIZE_OVERRIDE / -DMAX_GENERATIONS_OVERRIDE). D (the
// number of fitness cases) is a runtime argument so the harness can sweep it
// without recompiling.
//
//   usage:  ./cuda_evo_bench [D]        (D defaults to 32)
//   prints CSV rows (no header):  strategy,pop,D,gens,mean_size,seconds,gpops_s,best_mse
// =============================================================================
#include "LGPEngine.h"
#include "EvaluatorCuda.h"
#include "Dataset.h"
#include <fstream>
#include <algorithm>
#include "LGPConfig.h"
#include "cuda_errors.h"
#include <memory>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

static void run(const char* name, CudaEvaluator::Strategy strat, const Dataset& ds,
                uint32_t seed, const Dataset* test) {
    // GALOP_EFF_LOG=<path>: run with intron-removed storage (bitwise-identical
    // fitness) and dump the per-generation footprint {Sigma L, Sigma L_eff}.
    const char* efflog = std::getenv("GALOP_EFF_LOG");
    auto layout = efflog ? CudaEvaluator::Layout::PackedEffective
                         : CudaEvaluator::Layout::Packed;
    auto eval = std::make_unique<CudaEvaluator>(strat, 256, layout);
    CudaEvaluator* evalp = eval.get();
    LGPEngine engine(seed, std::move(eval));
    const auto t0 = std::chrono::high_resolution_clock::now();
    engine.evolve_sr(ds);
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();

    if (efflog && strat == CudaEvaluator::Strategy::Warp) {
        FILE* f = fopen(efflog, "w");
        fprintf(f, "generation,sum_L,sum_L_eff\n");
        int g = 0;
        for (const auto& [L, Leff] : evalp->footprint_log())
            fprintf(f, "%d,%lld,%lld\n", g++, L, Leff);
        fclose(f);
    }

    double ops = 0.0, sbar_sum = 0.0;
    for (const auto& s : engine.history_view()) {
        ops      += (double)LGPConfig::POPULATION_SIZE * s.mean_length * ds.N;  // P · S_bar_g · D
        sbar_sum += s.mean_length;
    }
    const int    G       = (int)engine.history_view().size();
    const double gpops_s = ops / sec;                 // RAW operations per second
    const double sbar    = G ? sbar_sum / G : 0.0;

    if (test)
        printf("%s,%d,%d,%d,%.2f,%.4f,%.3e,%.9e,%.9e\n",
               name, LGPConfig::POPULATION_SIZE, ds.N, G, sbar, sec, gpops_s,
               engine.best_train_mse(), engine.best_test_mse(*test));
    else
        printf("%s,%d,%d,%d,%.2f,%.4f,%.3e,%.9e\n",
               name, LGPConfig::POPULATION_SIZE, ds.N, G, sbar, sec, gpops_s,
               engine.best_train_mse());
    fflush(stdout);
    // GALOP_BEST_LOG=path : append the decoded best evolved program (for the
    // qualitative "what did it discover" figure). Header line = the CSV row.
    if (const char* bp = std::getenv("GALOP_BEST_LOG")) {
        std::ofstream bf(bp, std::ios::app);
        bf << "# " << name << " P=" << LGPConfig::POPULATION_SIZE
           << " D=" << ds.N << " G=" << engine.history_view().size()
           << " train_mse=" << engine.best_train_mse();
        if (test) bf << " test_mse=" << engine.best_test_mse(*test);
        bf << "\n";
        engine.print_best_program(bf);
        bf << "\n";
    }
    fprintf(stderr, "BREAKDOWN %s: total=%.3fs eval=%.3fs (%.1f%%) host+other=%.3fs\n",
            name, sec, evalp->eval_seconds(), 100.0 * evalp->eval_seconds() / sec,
            sec - evalp->eval_seconds());
}

int main(int argc, char** argv) {
    // argv[1]: either a D (synthesize nguyen1 at that size) or a .csv path.
    // argv[2..]: optional engine seed (integer) and/or "warp" (warp-only), any order.
    Dataset ds;
    const std::string arg1 = (argc > 1) ? argv[1] : "32";
    auto load_any = [](const std::string& p) {
        // header column count decides the loader: 2 -> 1-D, more -> multi-feature
        std::ifstream f(p); std::string hdr; std::getline(f, hdr);
        const long ncols = 1 + std::count(hdr.begin(), hdr.end(), ',');
        return ncols > 2 ? load_csv_multi(p) : load_csv_1d(p);
    };
    if (arg1.find(".csv") != std::string::npos) ds = load_any(arg1);
    else ds = make_sr_dataset_1d(std::atoi(arg1.c_str()), -1.f, 1.f, Nguyen::Nguyen_1, 1234);

    uint32_t seed = 42;          // historical default (matches all prior runs)
    bool warp_only = false;
    Dataset test; bool have_test = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "warp") warp_only = true;
        else if (a.find(".csv") != std::string::npos) { test = load_any(a); have_test = true; }
        else seed = static_cast<uint32_t>(std::atoi(a.c_str()));
    }
    run("warp",  CudaEvaluator::Strategy::Warp,  ds, seed, have_test ? &test : nullptr);
    if (!warp_only) run("block", CudaEvaluator::Strategy::Block, ds, seed, have_test ? &test : nullptr);
    return 0;
}
