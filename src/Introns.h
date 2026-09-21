#ifndef INTRONS_H
#define INTRONS_H
#include "ISA.h"
#include "LGPConfig.h"
#include <cstdint>

// =============================================================================
// Structural-intron analysis for LGP programs (Brameier & Banzhaf).
//
// A program's output is r0 after the last instruction. Because LGP here is
// straight-line code (no branches), a single backward liveness pass computes
// EXACTLY which instructions can influence r0 -- not conservatively. Executing
// only the effective subset produces a BITWISE-identical r0: the effective
// instructions read the same values in the same order, and introns only write
// registers that are dead at that point.
//
// Backward pass, live = {r0}:
//   - if dest not live  -> intron (live set unchanged)
//   - if dest live      -> effective; kill dest, then mark reads live:
//         SIN/COS read only src2 (src1 ignored by apply_op)
//         all other ops read src1 and src2
//         src2 in constant mode reads no register
// (kill-then-mark order makes r0 = r0 + r1 keep r0 live via src1.)
// =============================================================================
namespace Introns {

// mask[i] = true iff instruction i is effective. Returns the effective count.
inline int effective_mask(const uint32_t* prog, int len, char* mask) {
    bool live[LGPConfig::NUM_REGISTERS] = {false};
    live[LGPConfig::OUTPUT_REG] = true;    // output register
    int cnt = 0;
    for (int i = len - 1; i >= 0; --i) {
        const uint32_t ins = prog[i];
        const uint8_t dest = ISA::get_dest_index(ins);
        if (!live[dest]) { mask[i] = 0; continue; }
        mask[i] = 1; ++cnt;
        live[dest] = false;                 // killed by this write...
        const uint8_t op = ISA::get_op(ins);
        const bool unary = (op == ISA::SIN || op == ISA::COS);
        if (!unary)
            live[ISA::get_src1_index(ins)] = true;   // ...then reads go live
        if (!ISA::is_src2_constant(ins))
            live[ISA::get_src2_index(ins)] = true;
    }
    return cnt;
}

// Copy only the effective instructions of prog into out (order preserved).
// Returns the effective length.
inline int extract_effective(const uint32_t* prog, int len, uint32_t* out) {
    char mask[LGPConfig::MAX_PROGRAM_SIZE];
    const int cnt = effective_mask(prog, len, mask);
    int k = 0;
    for (int i = 0; i < len; ++i)
        if (mask[i]) out[k++] = prog[i];
    return cnt;
}

}  // namespace Introns
#endif
