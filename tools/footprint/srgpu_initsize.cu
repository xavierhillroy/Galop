// Initial-population size probe for SRGPU: construct the engine exactly as srgpu_bench.cu does,
// run only its population initializer (CPU fitness), and average program length/depth.
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>
#define private public
#include "regression.cuh"
#undef private
using namespace std;
using namespace cusr;
int main(int argc, char** argv) {
    int P = argc > 1 ? atoi(argv[1]) : 20000;
    int hi = argc > 2 ? atoi(argv[2]) : 7;       // init_depth upper bound
    int maxd = argc > 3 ? atoi(argv[3]) : 7;     // max_program_depth (SRGPU_DEPTH)
    // D=64 quartic dataset, as in the ladder
    vector<vector<float>> X(64, vector<float>(1)); vector<float> y(64);
    srand(1234);
    for (int i = 0; i < 64; ++i) { float x = -1.f + 2.f * (rand() / (float)RAND_MAX); X[i][0] = x; y[i] = x*x*x*x + x*x*x + x*x + x; }
    RegressionEngine reg;
    reg.function_set = {ADD, SUB, MUL, DIV, SIN, COS};
    reg.use_gpu = false;
    reg.population_size = P;
    reg.generations = 1;
    reg.tournament_size = 3;
    reg.metric = metric_t::mean_square_error;
    reg.parsimony_coefficient = 0;
    reg.max_program_depth = maxd;
    reg.const_range = {-5, 5};
    reg.init_depth = {4, hi};
    reg.init_method = init_t::half_and_half;
    reg.dataset = X; reg.label = y; reg.variable_nums = 1;
    reg.do_population_init();
    double sl = 0, sd = 0; int maxl = 0, minl = 1 << 30;
    for (auto& p : reg.population) { sl += p.length; sd += p.depth; if (p.length > maxl) maxl = p.length; if (p.length < minl) minl = p.length; }
    printf("init_depth {4,%d} max_program_depth %d P=%d: mean length %.1f nodes (min %d, max %d), mean depth %.2f, sizeof(Node)=%zu B -> %.0f B host per program\n",
           hi, maxd, P, sl / P, minl, maxl, sd / P, sizeof(Node), sl / P * sizeof(Node));
    return 0;
}
