// srgpu_bench.cu — GALOP-campaign adapter for cusr (Zhang/Lensen/Sun, PRICAI'22).
// Reads our CSV datasets (last column = target), runs cusr at parity knobs
// (tournament 3, MSE metric, +,-,*,/,sin,cos; other params = authors' example
// defaults), scores best program on train AND held-out test via the library's
// own calculate_fitness_cpu. No seed control exists upstream (random_device);
// each invocation is an independent run.
//   usage: srgpu_bench <train.csv> <test.csv> <P> <G>
//   stdout: P,G,wall_s,train_mse,test_mse
#include "include/cusr.h"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;
using namespace cusr;

static void load_csv(const string &path, vector<vector<float>> &X, vector<float> &y) {
    ifstream f(path);
    if (!f) { cerr << "cannot open " << path << endl; exit(2); }
    string line; getline(f, line);            // header
    while (getline(f, line)) {
        if (line.empty()) continue;
        vector<float> row; stringstream ss(line); string cell;
        while (getline(ss, cell, ',')) row.push_back(stof(cell));
        y.push_back(row.back()); row.pop_back();
        X.push_back(row);
    }
}

int main(int argc, char **argv) {
    if (argc < 5) { cerr << "usage: srgpu_bench train test P G\n"; return 2; }
    vector<vector<float>> Xtr, Xte; vector<float> ytr, yte;
    load_csv(argv[1], Xtr, ytr); load_csv(argv[2], Xte, yte);

    RegressionEngine reg;
    reg.function_set = {ADD, SUB, MUL, DIV, SIN, COS};   // parity primitive set
    reg.use_gpu = true;
    reg.population_size = atoi(argv[3]);
    reg.generations = atoi(argv[4]);
    reg.tournament_size = 3;                              // parity
    reg.metric = metric_t::mean_square_error;             // parity objective
    reg.parsimony_coefficient = 0;
    // authors' default 10; SRGPU_DEPTH env overrides (grid arm: 7 = parity
    // capacity, <=127 nodes vs the 101-node caps; legacy anchors keep 10)
    const char* dep = std::getenv("SRGPU_DEPTH");
    reg.max_program_depth = dep ? std::atoi(dep) : 10;
    // PARITY (set by us; their default is {-1,1}, regression.cuh:44). Their
    // constants are uniform ERCs over a range -- a discrete pool is not
    // expressible -- so we widen to cover the shared pool's unit-scale core
    // (0, +-1, pi, 0.5, 2, +-5, ...), which their default cannot sample.
    // Wider still (to reach 10..1000) would dilute density near the small
    // constants these targets need. Strict superset of their default:
    // direction favours SRGPU.
    reg.const_range = {-5, 5};
    // Initial-depth range. Lower bound: their default 4. Upper bound: the depth
    // cap (SRGPU_DEPTH) or their default 8, whichever is smaller -- the library
    // enforces max_program_depth only on offspring (hoist inside do_mutation),
    // never at initialization, so an upper bound above the cap seeds generation
    // 1 with over-cap trees. SRGPU_INIT_DEPTH_HI overrides: the recorded grid and
    // quality arms used 10 (their README example) and export it so they reproduce
    // exactly as run.
    const char* ih = std::getenv("SRGPU_INIT_DEPTH_HI");
    const int init_hi = ih ? std::atoi(ih) : (reg.max_program_depth < 8 ? reg.max_program_depth : 8);
    reg.init_depth = {4, init_hi};
    reg.init_method = init_t::half_and_half;
    reg.fit(Xtr, ytr);

    program::calculate_fitness_cpu(&reg.best_program, Xtr, ytr, (int)Xtr.size(), metric_t::mean_square_error);
    const float train_mse = reg.best_program.fitness;
    program::calculate_fitness_cpu(&reg.best_program, Xte, yte, (int)Xte.size(), metric_t::mean_square_error);
    const float test_mse = reg.best_program.fitness;
    printf("%d,%d,%.4f,%.6e,%.6e\n", reg.population_size, reg.generations,
           reg.regress_time_in_sec, train_mse, test_mse);
    fprintf(stderr, "best: %s\n", program::prefix_to_infix(reg.best_program.prefix).c_str());
    return 0;
}
