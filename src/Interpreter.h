#ifndef INTERPRETER_H
#define INTERPRETER_H
#include "LGPConfig.h"
#include <cstdint>

namespace Interpreter{
    // Run stateless one program across NUM_CONTEXTS parallel contexts.
//
// instructions:  pointer to the program's flat instruction array
// inst_length:        number of active instructions in the program (assumes max is 255)
// inputs:        layout: inputs[c * num_inputs + i] = input i of context c
//                where 0 <= c < NUM_CONTEXTS and 0 <= i < num_inputs
// num_inputs:    number of input values per context. Must be <= NUM_REGISTERS.
// outputs:       layout: outputs[c] = r0 of context c after execution.
//                 Size: NUM_CONTEXTS.
// Convention: r0..r_{num_inputs-1} are initialized with inputs.
//             r_{num_inputs}..r_{NUM_REGISTERS-1} start at 0.
//             Output of each context is r0 after the last instruction.

void run_stateless(const uint32_t* instructions, int inst_length, const float* inputs, int num_inputs, float* outputs );

    // Run statefull one program across NUM_CONTEXTS parallel contexts. (parallel in GPU sequential in CPU )
//
// instructions:  pointer to the program's flat instruction array
// inst_length:        number of active instructions in the program (assumes max is 255)
// inputs:        layout: inputs[c * num_inputs + i] = input i of context c
//                where 0 <= c < NUM_CONTEXTS and 0 <= i < num_inputs
// num_inputs:    number of input values per context. Must be <= NUM_REGISTERS.
// registers:     pointer to persistent registers  Shape [c *NUM_REGISTERS +reg] = reg for context c 
// outputs:       layout: outputs[c] = r0 of context c after execution.
//                Size: NUM_CONTEXTS.
// Convention: r0..r_{num_inputs-1} are initialized with inputs.
//             r_{num_inputs}..r_{NUM_REGISTERS-1} start at 0.
//             Output of each context is r0 after the last instruction.
// NOT IMPLEMENTED YET
void run_stateful(const uint32_t* instructions, int inst_length, const float* inputs, int num_inputs,float* registers, float* outputs);
}


#endif