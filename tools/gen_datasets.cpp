// gen_datasets.cpp
#include "Dataset.h"
#include <string>

struct Spec { const char* name; float (*fn)(float); float lo; float hi; };

int main() {
    const std::vector<Spec> specs = {
        {"nguyen1", Nguyen::Nguyen_1, -1.f, 1.f},
        {"nguyen2", Nguyen::Nguyen_2, -1.f, 1.f},
        {"nguyen3", Nguyen::Nguyen_3, -1.f, 1.f},
        {"nguyen4", Nguyen::Nguyen_4, -1.f, 1.f},
        {"nguyen5", Nguyen::Nguyen_5, -1.f, 1.f},
        {"nguyen6", Nguyen::Nguyen_6, -1.f, 1.f},
    };

    const uint32_t TRAIN_SEED = 1234;   // fixed, shared across all targets
    const uint32_t TEST_SEED  = 5678;   // different draw, same domain
    const int TRAIN_N = 32;
    const int TEST_N  = 100;

    for (const auto& s : specs) {
        const std::string base = std::string("datasets/") + s.name;
        write_csv(make_sr_dataset_1d(TRAIN_N, s.lo, s.hi, s.fn, TRAIN_SEED),
                  base + "_train.csv");
        write_csv(make_sr_dataset_1d(TEST_N,  s.lo, s.hi, s.fn, TEST_SEED),
                  base + "_test.csv");
    }
}