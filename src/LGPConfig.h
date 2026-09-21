#ifndef LGP_CONFIG_H
#define LGP_CONFIG_H

#include <cstdint>
#include <array>

// =============================================================================
// Named configuration bundles (must precede the individual override blocks).
//
//   -DGALOP_STANDARD     : the paper's standard configuration (§3.5) —
//                          inputs read-only + 3 insert attempts per child.
//                          Registers stay per-arity: add
//                          -DNUM_REGISTERS_OVERRIDE=16 for multi-input
//                          problems (8-register default covers 1-input).
//   -DGALOP_STANDARD_EM  : the one named variant — STANDARD + effmut.
//
// Bundles only #define the individual overrides (guarded, so explicit flags
// still win); without a bundle flag, nothing changes and the defaults remain
// byte-identical to the golden baseline.
// =============================================================================
#ifdef GALOP_STANDARD_EM
  #ifndef GALOP_STANDARD
    #define GALOP_STANDARD 1
  #endif
  #ifndef EFFMUT_OVERRIDE
    #define EFFMUT_OVERRIDE 1
  #endif
#endif
#ifdef GALOP_STANDARD
  #ifndef INPUT_PROTECT_OVERRIDE
    #define INPUT_PROTECT_OVERRIDE 1
  #endif
  #ifndef MACRO_INSERTS_OVERRIDE
    #define MACRO_INSERTS_OVERRIDE 3
  #endif
#endif

// =============================================================================
// LGPConfig: Compile-time configuration for the LGP system.
//
// All values here are `constexpr` -- they are baked into the binary at compile
// time. Changing any of these values requires a rebuild, but in exchange the
// compiler can use them for sizing, masking, and loop unrolling, and the CPU
// pays no runtime cost to read them.
//
// This file is intentionally parameter-only: it answers "how big is the
// population?", "how many registers?", "which constants can programs use?".
// The bit layout that turns these parameters into an instruction encoding
// lives in ISA.h.
// =============================================================================

namespace LGPConfig {

    // -------------------------------------------------------------------------
    // Utility: compile-time integer log2.
    // Used to derive bit-widths from pool sizes (e.g. 8 registers -> 3 bits).
    // Recursive because C++14 constexpr functions can't use loops portably;
    // by C++17 we could rewrite as a loop, but this is fine and runs at
    // compile time only.
    // -------------------------------------------------------------------------
    constexpr int log2(int n) {
        return (n <= 1) ? 0 : 1 + log2(n / 2);
    }

    // =========================================================================
    // Evolutionary loop parameters
    // =========================================================================

    // Override via -DMAX_GENERATIONS_OVERRIDE=N (benchmarks). Default unchanged.
#ifdef MAX_GENERATIONS_OVERRIDE
    constexpr int MAX_GENERATIONS = MAX_GENERATIONS_OVERRIDE;
#else
    constexpr int MAX_GENERATIONS = 1000;  // Outer loop cap on the evolutionary run.
#endif

    // =========================================================================
    // Population and program sizing
    //
    // Memory layout: all programs are stored as one flat uint32_t buffer of
    // length TOTAL_INSTRUCTIONS. Program i occupies the slice
    //   [i * MAX_PROGRAM_SIZE, i * MAX_PROGRAM_SIZE + program_lengths[i])
    // Slots past program_lengths[i] exist but are not executed by the
    // interpreter. This "program-major" layout is warp-friendly: on the GPU,
    // one warp evaluating one program reads its instruction stream
    // contiguously.
    // =========================================================================

    // Override via -DPOPULATION_SIZE_OVERRIDE=N (benchmarks). Default unchanged.
#ifdef POPULATION_SIZE_OVERRIDE
    constexpr int POPULATION_SIZE = POPULATION_SIZE_OVERRIDE;
#else
    constexpr int POPULATION_SIZE = 1000;    // Number of programs per generation. MUST BE MULTIPLE OF 2...
#endif
    static_assert((POPULATION_SIZE%2) == 0, "POPULATION_SIZE must be mult of 2");
                                                 // Small for now while testing;
                                                 // will be bumped up later.
    // Override via -DMAX_PROGRAM_SIZE_OVERRIDE=100 (GALOP-tuned arms; breaks the
    // 50-instr ~ 101-tree-node op-parity with EvoGP -- label such runs "tuned").
#ifdef MAX_PROGRAM_SIZE_OVERRIDE
    constexpr int MAX_PROGRAM_SIZE = MAX_PROGRAM_SIZE_OVERRIDE;
    static_assert(MAX_PROGRAM_SIZE <= 254, "uint8 lengths; 255 is the sentinel");
#else
    constexpr int MAX_PROGRAM_SIZE = 50;   // Hard cap on instructions per program. MAX MAX FOR NOW IS 255 because using uint8 att some places
#endif
                                                 // All programs reserve this much
                                                 // space; variable length is tracked
                                                 // separately in program_lengths[].
    constexpr int SIZE_SENTINEL = 255; // max size for uint8 acts as a flag for invalid length 
    // Override via -DSTARTING_PROGRAM_SIZE_OVERRIDE=20 (bigger-programs arms).
#ifdef STARTING_PROGRAM_SIZE_OVERRIDE
    constexpr int STARTING_PROGRAM_SIZE = STARTING_PROGRAM_SIZE_OVERRIDE;
#else
    constexpr int STARTING_PROGRAM_SIZE = 8;    // Initial length of every program
#endif
                                                 // at generation 0.
    constexpr long long TOTAL_INSTRUCTIONS =
        (long long)POPULATION_SIZE * MAX_PROGRAM_SIZE;   // > int32 at P ~ 43M
                                                 // Total slots in the flat buffer.

    // =========================================================================
    // Register file
    //
    // Each program has its own set of scalar float registers. Register IDs
    // are packed into the instruction word, so NUM_REGISTERS must be a power
    // of 2 and must fit in the index field (currently 7 bits -> max 128).
    // =========================================================================
    // IMPORTANT MASK SIZE WILL BASICALLY SET THE BITS ACTUALLY USED TO REPRESENT THE NUM REG IN BINARY 
    // Override via -DNUM_REGISTERS_OVERRIDE=16 (multi-feature datasets: inputs
    // occupy r0..k-1, so 8-feature problems need scratch beyond 8 registers).
    // Default unchanged; the constant pool below grows in lockstep (the ISA
    // requires NUM_CONSTANTS == NUM_REGISTERS).
#ifdef NUM_REGISTERS_OVERRIDE
    constexpr int NUM_REGISTERS = NUM_REGISTERS_OVERRIDE;  // MUST be a power of 2.
#else
    constexpr int NUM_REGISTERS = 8;                   // MUST be a power of 2.
#endif
    constexpr int REGISTER_BITS = log2(NUM_REGISTERS); // Bits needed to encode an ID.
    constexpr int REGISTER_MASK = NUM_REGISTERS - 1;   // AND-mask that clips any value
                                                        // to a valid register index.
                                                        // For NUM_REGISTERS=8 this is 0b111.

    // =========================================================================
    // Constant pool
    //
    // When an instruction's src2 field is flagged as "constant mode", its
    // index selects into CONSTANTS[] instead of the register file. These are
    // fixed at compile time -- programs cannot evolve new constants, only
    // pick from this list.
    // =========================================================================
    //
    constexpr int NUM_CONSTANTS = NUM_REGISTERS;
    constexpr int CONSTANT_BITS = log2(NUM_CONSTANTS);
    constexpr int CONSTANT_MASK = NUM_CONSTANTS - 1; // produces 7 which is 111 


    // First 8 entries are the historical pool (default builds are unchanged);
    // the extension entries exist only under NUM_REGISTERS_OVERRIDE=16 and must
    // be mirrored in EvoGP's const_samples for parity (CONFIG_PARITY.md).
#if defined(NUM_REGISTERS_OVERRIDE) && NUM_REGISTERS_OVERRIDE == 16
    constexpr auto CONSTANTS = std::to_array<float>({
    0.0f, 1.0f, -1.0f, 3.14159f, 0.5f, 2.0f, 10.0f, 100.0f,
    0.1f, -0.5f, 0.25f, 5.0f, 20.0f, -2.0f, 0.01f, 1000.0f
    });
#else
    constexpr auto CONSTANTS = std::to_array<float>({
    0.0f, 1.0f, -1.0f, 3.14159f, 0.5f, 2.0f, 10.0f, 100.0f
    });
#endif
    static_assert(CONSTANTS.size() == NUM_CONSTANTS,
              "CONSTANTS pool size doesn't match NUM_CONSTANTS");

    // The src2 index field in an instruction is 7 bits wide and is used for
    // BOTH registers (when mode=0) and constants (when mode=1). Keeping the
    // two pools equal-sized means one mask suffices for both cases. If you
    // ever want asymmetric pools, this assumption breaks and the encoder/
    // decoder need reworking -- the static_assert will catch it at compile
    // time.
    static_assert(NUM_REGISTERS == NUM_CONSTANTS,
                  "src2 index field assumes NUM_REGISTERS == NUM_CONSTANTS");

    // =========================================================================
    // Instruction set size
    //
    // The opcode occupies the low byte of the instruction word. Only
    // OPERATIONS_BITS of that byte are meaningful; the rest must be masked.
    // The concrete opcode enum lives in ISA.h (ADD, SUB, MUL, ...).
    // =========================================================================

    constexpr int NUM_OPERATIONS = 8;                    // Max 2^8 = 256 if we used the whole byte.
    constexpr int OPERATIONS_BITS = log2(NUM_OPERATIONS); // Active bits in the opcode field.
    constexpr int OPERATION_MASK = NUM_OPERATIONS - 1;   // AND-mask that clips a random byte
                                                           // to a valid opcode index.
                                                           // Example: 8 ops -> mask = 0b0000_0111.

    // =========================================================================
    // Variation rates
    //
    // These control the probabilities applied during mutation and crossover.
    // Names correspond to specific operators that will be implemented later;
    // documenting semantics here as placeholders.
    // =========================================================================

    // P(a macro-INSERT happens at all), applied MACRO_INSERTS times per child.
    // (Where the insert lands -- tail vs random position -- is MACRO_RANDOM_POS,
    // not this rate.) Override via -DINSERT_TAIL_RATE_OVERRIDE=0.4f.
#ifdef INSERT_TAIL_RATE_OVERRIDE
    constexpr float INSERT_TAIL_RATE = INSERT_TAIL_RATE_OVERRIDE;
#else
    constexpr float INSERT_TAIL_RATE = 0.8f;
#endif
    constexpr float DELETE_TAIL_RATE = 0.75f;  // P(a macro-DELETE happens) per child.
    // Override via -DMICRO_RATE_OVERRIDE=0.1f (operator-tuning sweeps). Default unchanged.
#ifdef MICRO_RATE_OVERRIDE
    constexpr float MICRO_RATE = MICRO_RATE_OVERRIDE;
#else
    constexpr float MICRO_RATE = 0.8f;   // P(apply micro-mutation PER CHILD -- one field of one
                                          // instruction), not per-instruction.
#endif

    // --- operator-tuning flags (compile-time; defaults preserve historical
    //     behavior so the golden stays valid) --------------------------------
    // Random-position insert/delete instead of tail-only. Tail-only was chosen
    // to avoid mid-array shifts, but at L<=50 (<=200 B) a memmove is ~ns;
    // search-wise, mid-program insertion grows subexpressions that tail
    // insertion cannot.
// ADOPTED 2026-07: random-position insert/delete is now the default. It closes
// the SR quality gap vs EvoGP (5/6 no-sig-diff, matched-P AND iso-time; was
// 5/6 EvoGP-better with tail-only) at ~15% host-op cost. Force the historical
// tail-only behavior with -DMACRO_TAIL_ONLY_OVERRIDE for A/B comparison.
#ifdef MACRO_TAIL_ONLY_OVERRIDE
    constexpr bool MACRO_RANDOM_POS = false;
#else
    constexpr bool MACRO_RANDOM_POS = true;
#endif
    // Per-INSTRUCTION micro-mutation probability (standard LGP semantics,
    // expected mutations scale with length). 0 = off -> historical per-CHILD
    // single micro-mutation gated by MICRO_RATE.
#ifdef MICRO_PER_INSTR_OVERRIDE
    constexpr float MICRO_PER_INSTR = MICRO_PER_INSTR_OVERRIDE;
#else
    constexpr float MICRO_PER_INSTR = 0.0f;
#endif

    constexpr float CROSSOVER_RATE = 1.0f;   // P(apply crossover) per offspring.
    // Override via -DELITES_OVERRIDE=0.05f (selection-pressure sweeps). Default unchanged.
#ifdef ELITES_OVERRIDE
    constexpr float ELITES = ELITES_OVERRIDE;
#else
    constexpr float ELITES = 0.1f;   // Fraction of population preserved as elites.
#endif
    constexpr int ELITE_COUNT = static_cast<int>(LGPConfig::POPULATION_SIZE * LGPConfig::ELITES);
    constexpr int NON_ELITE_COUNT = LGPConfig::POPULATION_SIZE - ELITE_COUNT;
    static_assert(NON_ELITE_COUNT % 2 == 0,
              "Non-elite slots must be even -- vary_pair produces two children");
    // Override via -DTOURNAMENT_SIZE_OVERRIDE=5 (selection-pressure sweeps). Default unchanged.
#ifdef TOURNAMENT_SIZE_OVERRIDE
    constexpr int TOURNAMENT_SIZE = TOURNAMENT_SIZE_OVERRIDE;
#else
    constexpr int TOURNAMENT_SIZE = 3;
#endif

    // =========================================================================
    // Search-efficiency levers (all default-off; quality-tuning arms)
    // =========================================================================
    // EFFMUT: bias variation onto structurally effective code (Brameier &
    // Banzhaf's "effective mutation"). With ~90% introns, unbiased position
    // choice wastes ~9/10 variation events on dead code; this fixes the
    // per-child effective-variation rate at ~1. -DEFFMUT_OVERRIDE=1 to enable.
#ifdef EFFMUT_OVERRIDE
    constexpr bool EFFMUT = true;
#else
    constexpr bool EFFMUT = false;
#endif
    // Blend: probability that a given variation event uses the effective-code
    // bias (else uniform -> neutral intron drift preserved). 1.0 = pure effmut.
    // -DEFFMUT_RATE_OVERRIDE=0.5f for the blended arm.
#ifdef EFFMUT_RATE_OVERRIDE
    constexpr float EFFMUT_RATE = EFFMUT_RATE_OVERRIDE;
#else
    constexpr float EFFMUT_RATE = 1.0f;
#endif
    // Macro-insert attempts per child (each at INSERT_TAIL_RATE).
    // -DMACRO_INSERTS_OVERRIDE=3 for the multi-insert arm.
#ifdef MACRO_INSERTS_OVERRIDE
    constexpr int MACRO_INSERTS = MACRO_INSERTS_OVERRIDE;
#else
    constexpr int MACRO_INSERTS = 1;
#endif
    // INPUT_PROTECT: write-protect input registers (dest restricted to the
    // upper half of the register file) and relocate the output register to the
    // first scratch register. Canonical LGP keeps inputs readable for the whole
    // program; without this, multi-feature programs destroy their own inputs.
    // -DINPUT_PROTECT_OVERRIDE=1 (use with NUM_REGISTERS_OVERRIDE=16).
#ifdef INPUT_PROTECT_OVERRIDE
    constexpr bool INPUT_PROTECT = true;
    constexpr int  OUTPUT_REG    = NUM_REGISTERS / 2;   // first scratch register
#else
    constexpr bool INPUT_PROTECT = false;
    constexpr int  OUTPUT_REG    = 0;
#endif


    // =========================================================================
    // RNG
    // =========================================================================

    constexpr uint32_t SEED = 42;  // Fixed seed -> deterministic runs for testing.

    constexpr int NUM_CONTEXTS = 32; // number of contexts / environments...

}  // namespace LGPConfig

#endif  // LGP_CONFIG_H