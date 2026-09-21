// EvaluatorCpu.cpp
#include "EvaluatorCpu.h"
#include "Evaluator.h"
#include "LGPConfig.h"
#include <cmath>

void CpuEvaluator::evaluate_population(const uint32_t* instructions, const uint8_t* lengths,
                                       float* fitness_out, int n_programs, const Dataset& ds) {
    for (int i = 0; i < n_programs; ++i) {
                         // same NaN-skip built in by caller ( doesnt pass in ELITES)
            ProgramView prog{ instructions + (size_t)i * LGPConfig::MAX_PROGRAM_SIZE,
                              static_cast<int>(lengths[i]) };
            fitness_out[i] = Fitness::mse_to_fitness(Evaluator::evaluate_sr_mse(prog, ds));
        
    }
}