// gen_sweep_datasets.cpp — generate Nguyen datasets at swept D (data points).
//
// Writes to datasets/sweep/ so the golden datasets (datasets/nguyen*_train.csv)
// are untouched. These files are what EvoGP will read, so both systems evaluate
// on IDENTICAL data. GALOP synthesizes the same points in-memory via the same
// make_sr_dataset_1d(D, ..., seed) call, so the data matches by construction.
//
//   - crossover: nguyen1 at every D in the sweep (train only; timing/throughput)
//   - quality:   all 6 Nguyen at a fixed train/test D (for the MSE table)
//
// D list: pass on the command line, or use the default geometric sweep 4..131072.
#include "Dataset.h"
#include <string>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <cstdio>

int main(int argc, char** argv) {
    std::vector<int> Ds;
    if (argc > 1) for (int i = 1; i < argc; ++i) Ds.push_back(std::atoi(argv[i]));
    else Ds = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
               4096, 8192, 16384, 32768, 65536, 131072};

    const uint32_t TRAIN_SEED = 1234, TEST_SEED = 5678;   // must match GALOP's synth seed
    std::filesystem::create_directories("datasets/sweep");

    // --- crossover: nguyen1 across the D sweep (train only) ---
    for (int D : Ds) {
        Dataset tr = make_sr_dataset_1d(D, -1.f, 1.f, Nguyen::Nguyen_1, TRAIN_SEED);
        write_csv(tr, "datasets/sweep/nguyen1_D" + std::to_string(D) + "_train.csv");
    }
    printf("crossover: nguyen1 at %zu D values (4..%d)\n", Ds.size(), Ds.back());

    // --- quality: all 6 Nguyen at a fixed train/test size ---
    struct Spec { const char* name; float (*fn)(float); };
    const std::vector<Spec> specs = {
        {"nguyen1", Nguyen::Nguyen_1}, {"nguyen2", Nguyen::Nguyen_2},
        {"nguyen3", Nguyen::Nguyen_3}, {"nguyen4", Nguyen::Nguyen_4},
        {"nguyen5", Nguyen::Nguyen_5}, {"nguyen6", Nguyen::Nguyen_6},
    };
    const int TRAIN_N = 100, TEST_N = 200;
    for (const auto& s : specs) {
        const std::string base = std::string("datasets/sweep/") + s.name;
        write_csv(make_sr_dataset_1d(TRAIN_N, -1.f, 1.f, s.fn, TRAIN_SEED), base + "_train.csv");
        write_csv(make_sr_dataset_1d(TEST_N,  -1.f, 1.f, s.fn, TEST_SEED),  base + "_test.csv");
    }
    printf("quality: 6 Nguyen at train=%d test=%d\n", TRAIN_N, TEST_N);
    return 0;
}
