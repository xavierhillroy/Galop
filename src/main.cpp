// main.cpp — Phase 2 driver: runs ONE sweep cell (one target, one seed).
//
// usage: ./lgp_run <train_csv> <test_csv> <target_name> <run_seed>
//                  <results_csv> <program_txt> <history_csv>
//
// Writes:
//   <results_csv> : appends one numeric row (target,tool,run_seed,train_mse,test_mse,best_length)
//   <program_txt> : the best program, human-readable, with a self-describing header
//   <history_csv> : per-generation convergence trace (one row per generation)
//
// One job = one invocation. Concatenate all <results_csv> files afterward
// and prepend the header once. Each results file is one row, no header.
// Each history file HAS its own header (it's a standalone multi-row trace);
// strip duplicate headers when concatenating histories (see note at bottom).
#include "LGPEngine.h"
#include "Dataset.h"
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: " << argv[0]
                  << " <train_csv> <test_csv> <target_name>"
                     " <run_seed> <results_csv> <program_txt> <history_csv>\n";
        return 1;
    }

    const std::string train_path  = argv[1];
    const std::string test_path   = argv[2];
    const std::string target_name = argv[3];
    const uint32_t    run_seed    = static_cast<uint32_t>(std::stoul(argv[4]));
    const std::string results_csv = argv[5];
    const std::string program_txt = argv[6];
    const std::string history_csv = argv[7];

    // Load the shared, fixed datasets. load_csv_1d throws on missing file,
    // so a bad path fails loudly rather than producing an empty dataset.
    Dataset train, test;
    try {
        train = load_csv_1d(train_path);
        test  = load_csv_1d(test_path);
    } catch (const std::exception& e) {
        std::cerr << "[" << target_name << " seed " << run_seed
                  << "] dataset load failed: " << e.what() << "\n";
        return 1;
    }

    // Run one evolutionary search with this cell's seed. Fresh engine =
    // fully clean state; the seed is the only thing that varies across cells.
    LGPEngine engine(run_seed);
    engine.evolve_sr(train);

    const float train_mse = engine.best_train_mse();      // floored at 0 (it's fitness)
    const float test_mse  = engine.best_test_mse(test);   // raw R^2, may be negative
    const int   best_len = engine.best_length();

    // Ensure output dirs exist (cluster jobs launch from arbitrary cwd, and
    // ofstream will NOT create parent directories -- it just fails silently).
    std::filesystem::path rp(results_csv), pp(program_txt), hp(history_csv);
    if (rp.has_parent_path()) std::filesystem::create_directories(rp.parent_path());
    if (pp.has_parent_path()) std::filesystem::create_directories(pp.parent_path());
    if (hp.has_parent_path()) std::filesystem::create_directories(hp.parent_path());

    // --- numeric result: one row, append mode, no header ---
    {
        std::ofstream f(results_csv, std::ios::app);
        if (!f) { std::cerr << "cannot open " << results_csv << "\n"; return 1; }
        f << target_name << ",lgp," << run_seed << ','
          << train_mse << ',' << test_mse << ',' << best_len << '\n';
    }

    // --- best program: its own file, self-describing header ---
    {
        std::ofstream f(program_txt);
        if (!f) { std::cerr << "cannot open " << program_txt << "\n"; return 1; }
        f << "# target=" << target_name
          << " seed="    << run_seed
          << " train_mse="<< train_mse
          << " test_mse=" << test_mse
          << " length="  << best_len << "\n";
        engine.print_best_program(f);   // requires the ostream& overload
    }

    // --- convergence history: one row per generation, with target/seed tags ---
    // The engine already records GenerationStats every generation into history;
    // this just serialises it. Tagged with target+seed so all history files
    // merge into one tidy table for the convergence plots.
    {
        std::ofstream f(history_csv);
        if (!f) { std::cerr << "cannot open " << history_csv << "\n"; return 1; }
        f << "target,run_seed,generation,best_fitness,mean_fitness,"
             "best_length,mean_length\n";
        for (const auto& s : engine.history_view()) {
            f << target_name << ',' << run_seed << ','
              << s.generation     << ','
              << s.best_fitness   << ','
              << s.mean_fitness   << ','
              << s.best_length    << ','
              << s.mean_length    << '\n';
        }
    }

    // A line to stdout so the SLURM .out log confirms the cell ran.
    std::cout << target_name << " seed " << run_seed
              << " : train_mse=" << train_mse
              << " test_mse="    << test_mse << "\n";
    return 0;
}