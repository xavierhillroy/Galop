#ifndef DATASET_H
#define DATASET_H
#include <vector>
#include <cstdint>
#include "LGPConfig.h"
#include <fstream>
// =============================================================================
// Dataset: inputs and targets for symbolic regression.
//
// Layout matches Interpreter::run_stateless:
//   inputs[n * num_inputs + i] = input i for fitness case n
//   targets[n]                  = ground truth for case n
//
// Stored row count is padded up to a multiple of NUM_CONTEXTS so the
// chunked evaluator can always issue a full 32-wide interpreter call
// without reading past the array end. Real case count is N. Padded rows
// are zero-filled and are never counted in the MSE sum -- the evaluator
// iterates only up to N and accumulates only the first `valid` outputs
// per chunk.
// =============================================================================

struct Dataset {
    std::vector<float> inputs; // Size = padded_N * num inputs (pads to context size (32 ;))- this is all to allign with warp level 
    std::vector<float> targets; /// size = padded_N
    int N; 
    int num_inputs; // num inputs of sr equation 
    // Variance of the target over the N valid points: SS_tot = Σ(y - ȳ)².
    // Constant per dataset, so computed once at construction and reused by
    // the R² evaluator instead of being recomputed per fitness call.
    // 0.0 means a degenerate (constant) target -> R² undefined.
    double ss_tot = 0.0;


    int padded_N() const {
        const int c = LGPConfig::NUM_CONTEXTS; // num contexts - padded to 32 for warp 
        return((N + c -1)/c)*c ;
    }
    

};
namespace SRTargets {
    float quadratic(float x); // x^2 + x 
    float koza1(float x); // x^4 + x^3 + x^2 + x 
    float identity(float x); //f(x) = x;
    float koza3(float x);

}
namespace Nguyen{
    float Nguyen_1(float x);
    float Nguyen_2(float x);
    float Nguyen_3(float x);
    float Nguyen_4(float x);
    float Nguyen_5(float x);
    float Nguyen_6(float x);


}
// build a dataset by sampling unifromaly from [x_min, x_max] for a single inpout target fn- determinist function used from config 
Dataset make_sr_dataset_1d(int N, float xmin, float xmax, float (*target_fn)(float),uint32_t seed);
Dataset load_csv_1d(const std::string& path);
// Multi-feature CSV: every column but the last is a feature, last is the
// target. Same padding/ss_tot conventions as the 1-D loader.
Dataset load_csv_multi(const std::string& path);
void write_csv(const Dataset& d, const std::string& path);
#endif
