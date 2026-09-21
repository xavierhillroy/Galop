#include "LGPEngine.h"
#include "LGPConfig.h"
#include "ISA.h"
#include "Interpreter.h"
#include "Introns.h"
#include <cstring>
#include "Dataset.h"
#include "Evaluator.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <vector>
#include "GenerationStats.h"
// =============================================================================
// Test harness
// =============================================================================
static int g_tests_run    = 0;
static int g_tests_passed = 0;

// std::endl (not "\n") forces a flush so the [ RUN ] line is visible before
// an assert() abort. do/while(0) keeps the macro a single statement.
#define RUN(test_fn) do {                                       \
    std::cout << "[ RUN  ] " #test_fn << std::endl;             \
    ++g_tests_run;                                              \
    test_fn();                                                  \
    ++g_tests_passed;                                           \
    std::cout << "[  OK  ] " #test_fn << std::endl;             \
} while(0)

// Helper used by an existing eyeball-print test path; kept around because
// it's handy for debugging a failing instruction-decode test.
void print_instruction(uint32_t instr) {
    std::cout << "op=" << (int)ISA::get_op(instr)
              << " dest=r" << (int)ISA::get_dest_index(instr)
              << " src1=r" << (int)ISA::get_src1_index(instr)
              << " src2=" << (ISA::is_src2_constant(instr) ? "C[" : "r[")
              << (int)ISA::get_src2_index(instr) << "]"
              << "\n";
}

// =============================================================================
// Engine init tests
// (pulled out of main() so they run through the same RUN harness as the rest)
// =============================================================================
void test_engine_init_invariants() {
    LGPEngine engine;
    engine.init_population();
    const PopulationData& data = engine.get_data();

    // Structural invariants
    assert(data.instructions_buf[0].size()    == LGPConfig::TOTAL_INSTRUCTIONS);
    assert(data.program_lengths_buf[0].size() == LGPConfig::POPULATION_SIZE);
    assert(data.fitness_scores_buf[0].size()  == LGPConfig::POPULATION_SIZE);
    assert(data.instructions_buf[0].size()    == LGPConfig::TOTAL_INSTRUCTIONS);
    assert(data.program_lengths_buf[0].size() == LGPConfig::POPULATION_SIZE);
    assert(data.fitness_scores_buf[0].size()  == LGPConfig::POPULATION_SIZE);

    // Every program should have the starting length
    for (auto len : data.program_lengths_buf[0]) {
        assert(len == LGPConfig::STARTING_PROGRAM_SIZE);
    }

    // Fitness should still be NaN sentinel (we haven't evaluated)
    for (auto f : data.fitness_scores_buf[0]) {
        assert(std::isnan(f));
    }
}

void test_engine_init_decoded_fields_valid() {
    LGPEngine engine;
    engine.init_population();
    const PopulationData& data = engine.get_data();

    // Every decoded instruction in the active range should have valid fields
    for (int p = 0; p < LGPConfig::POPULATION_SIZE; ++p) {
        int base = p * LGPConfig::MAX_PROGRAM_SIZE;
        for (int i = 0; i < data.program_lengths_buf[0][p]; ++i) {
            uint32_t instr = data.instructions_buf[0][base + i];
            assert(ISA::get_op(instr)         < LGPConfig::NUM_OPERATIONS);
            assert(ISA::get_dest_index(instr) < LGPConfig::NUM_REGISTERS);
            assert(ISA::get_src1_index(instr) < LGPConfig::NUM_REGISTERS);
            assert(ISA::get_src2_index(instr) < LGPConfig::NUM_REGISTERS);
        }
    }
}

// =============================================================================
// Interpreter tests
// =============================================================================
void test_interpreter_doubling() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    uint32_t prog[1] = {
        ISA::encode_manual(ISA::ADD, 0, 0, 0, false)
    };

    float inputs[LGPConfig::NUM_CONTEXTS];
    float outputs[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        inputs[c] = static_cast<float>(c);
    }

    Interpreter::run_stateless(prog, 1, inputs, 1, outputs);

    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        float expected = 2.0f * c;
        assert(outputs[c] == expected);
    }
}

void test_interpreter_zero_init() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // r0 = r0 + r1, where r1 is scratch (should be 0)
    uint32_t prog[1] = {
        ISA::encode_manual(ISA::ADD, 0, 0, 1, false)
    };

    float inputs[LGPConfig::NUM_CONTEXTS];
    float outputs[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        inputs[c] = static_cast<float>(c) + 1.0f;
    }

    Interpreter::run_stateless(prog, 1, inputs, 1, outputs);

    // r1 was zero, so output should equal input
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        assert(outputs[c] == inputs[c]);
    }
}

void test_interpreter_constant_access() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // r0 = r0 + C[1]   where C[1] = 1.0
    uint32_t prog[1] = {
        ISA::encode_manual(ISA::ADD, 0, 0, 1, /*is_constant=*/true)
    };

    float inputs[LGPConfig::NUM_CONTEXTS];
    float outputs[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        inputs[c] = static_cast<float>(c);
    }

    Interpreter::run_stateless(prog, 1, inputs, 1, outputs);

    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        float expected = static_cast<float>(c) + 1.0f;
        assert(outputs[c] == expected);
    }
}

void test_interpreter_subtract() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // r0 = r0 - C[1]    (C[1] = 1.0, so output = input - 1)
    uint32_t prog[1] = {
        ISA::encode_manual(ISA::SUB, 0, 0, 1, true)
    };
    float inputs[LGPConfig::NUM_CONTEXTS], outputs[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) inputs[c] = static_cast<float>(c);

    Interpreter::run_stateless(prog, 1, inputs, 1, outputs);

    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        assert(outputs[c] == static_cast<float>(c) - 1.0f);
    }
}

void test_interpreter_multiply() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // r0 = r0 * C[5]    (C[5] = 2.0)
    uint32_t prog[1] = {
        ISA::encode_manual(ISA::MUL, 0, 0, 5, true)
    };
    float inputs[LGPConfig::NUM_CONTEXTS], outputs[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) inputs[c] = static_cast<float>(c);

    Interpreter::run_stateless(prog, 1, inputs, 1, outputs);

    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        assert(outputs[c] == static_cast<float>(c) * 2.0f);
    }
}

void test_interpreter_protected_div() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // r0 = r0 / r1, where r1 is 0 (scratch).
    // Protected div returns 1.0 on zero divisor.
    uint32_t prog[1] = {
        ISA::encode_manual(ISA::DIV, 0, 0, 1, false)
    };
    float inputs[LGPConfig::NUM_CONTEXTS], outputs[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        inputs[c] = static_cast<float>(c) + 1.0f;  // any nonzero input
    }

    Interpreter::run_stateless(prog, 1, inputs, 1, outputs);

    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        assert(outputs[c] == 1.0f);  // protected-div fallback
        assert(!std::isnan(outputs[c]));
        assert(!std::isinf(outputs[c]));
    }
}

void test_interpreter_normal_div() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // r0 = r0 / C[5]   (C[5] = 2.0)
    uint32_t prog[1] = {
        ISA::encode_manual(ISA::DIV, 0, 0, 5, true)
    };
    float inputs[LGPConfig::NUM_CONTEXTS], outputs[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) inputs[c] = static_cast<float>(c) * 2.0f;

    Interpreter::run_stateless(prog, 1, inputs, 1, outputs);

    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        assert(outputs[c] == static_cast<float>(c));
    }
}

void test_interpreter_lt() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // r0 = r0 < C[5]   (C[5] = 2.0)
    uint32_t prog[1] = {
        ISA::encode_manual(ISA::LT, 0, 0, 5, true)
    };
    float inputs[LGPConfig::NUM_CONTEXTS], outputs[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) inputs[c] = static_cast<float>(c);

    Interpreter::run_stateless(prog, 1, inputs, 1, outputs);

    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        float expected = (static_cast<float>(c) < 2.0f) ? 1.0f : 0.0f;
        assert(outputs[c] == expected);
    }
}

void test_interpreter_gt() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // r0 = r0 > C[5]   (C[5] = 2.0)
    uint32_t prog[1] = {
        ISA::encode_manual(ISA::GT, 0, 0, 5, true)
    };
    float inputs[LGPConfig::NUM_CONTEXTS], outputs[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) inputs[c] = static_cast<float>(c);

    Interpreter::run_stateless(prog, 1, inputs, 1, outputs);

    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        float expected = (static_cast<float>(c) > 2.0f) ? 1.0f : 0.0f;
        assert(outputs[c] == expected);
    }
}

void test_interpreter_sin() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // r0 = sin(r0)
    uint32_t prog[1] = {
        ISA::encode_manual(ISA::SIN, 0, 0, 0, false)  // src2 ignored for unary
    };
    float inputs[LGPConfig::NUM_CONTEXTS], outputs[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) inputs[c] = static_cast<float>(c);

    Interpreter::run_stateless(prog, 1, inputs, 1, outputs);

    const float eps = 1e-5f;
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        float expected = std::sin(static_cast<float>(c));
        assert(std::fabs(outputs[c] - expected) < eps);
    }
}

void test_interpreter_multi_instruction() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // Program:
    //   r1 = r0 + C[1]    (r1 = x + 1)
    //   r0 = r1 * C[5]    (r0 = (x + 1) * 2)
    // For input x, output should be 2x + 2.
    uint32_t prog[2] = {
        ISA::encode_manual(ISA::ADD, 1, 0, 1, true),
        ISA::encode_manual(ISA::MUL, 0, 1, 5, true),
    };

    float inputs[LGPConfig::NUM_CONTEXTS], outputs[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) inputs[c] = static_cast<float>(c);

    Interpreter::run_stateless(prog, 2, inputs, 1, outputs);

    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        float expected = 2.0f * static_cast<float>(c) + 2.0f;
        assert(outputs[c] == expected);
    }
}

void test_interpreter_context_independence() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // r0 = r0 * r0    (each context squares its own input)
    uint32_t prog[1] = {
        ISA::encode_manual(ISA::MUL, 0, 0, 0, false)
    };

    float inputs[LGPConfig::NUM_CONTEXTS], outputs[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) inputs[c] = static_cast<float>(c);

    Interpreter::run_stateless(prog, 1, inputs, 1, outputs);

    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        float expected = static_cast<float>(c) * static_cast<float>(c);
        assert(outputs[c] == expected);
    }
}

void test_interpreter_determinism() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    uint32_t prog[1] = { ISA::encode_manual(ISA::ADD, 0, 0, 0, false) };
    float inputs[LGPConfig::NUM_CONTEXTS], out_a[LGPConfig::NUM_CONTEXTS], out_b[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) inputs[c] = static_cast<float>(c);

    Interpreter::run_stateless(prog, 1, inputs, 1, out_a);
    Interpreter::run_stateless(prog, 1, inputs, 1, out_b);

    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        assert(out_a[c] == out_b[c]);
    }
}

void test_interpreter_zero_length() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (default-decode semantics; INPUT_PROTECT build)\n", __func__);
        return;
    }
    uint32_t prog[1] = { 0 };  // never executed since length is 0
    float inputs[LGPConfig::NUM_CONTEXTS], outputs[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) inputs[c] = static_cast<float>(c) + 1.0f;

    Interpreter::run_stateless(prog, 0, inputs, 1, outputs);

    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) {
        assert(outputs[c] == inputs[c]);
    }
}

// =============================================================================
// Dataset tests
// =============================================================================
void test_dataset_basic_construction() {
    Dataset d = make_sr_dataset_1d(50, -1.0f, 1.0f, SRTargets::quadratic, 42);
    assert(d.N == 50);
    assert(d.num_inputs == 1);

    constexpr int C = LGPConfig::NUM_CONTEXTS;
    const int expected_pad = ((50 + C - 1) / C) * C;
    assert(d.padded_N()    == expected_pad);
    assert(d.inputs.size()  == static_cast<size_t>(expected_pad));
    assert(d.targets.size() == static_cast<size_t>(expected_pad));

    for (int n = 0; n < d.N; ++n) {
        assert(d.targets[n] == SRTargets::quadratic(d.inputs[n]));
    }
    for (int n = d.N; n < d.padded_N(); ++n) {
        assert(d.inputs[n]  == 0.0f);
        assert(d.targets[n] == 0.0f);
    }
}
void test_dataset_padding_boundaries() {
    constexpr int C = LGPConfig::NUM_CONTEXTS;

    // Exact multiple, just over, just under, single row.
    assert(make_sr_dataset_1d(C,     -1, 1, SRTargets::quadratic, 1).padded_N() == C);
    assert(make_sr_dataset_1d(C + 1, -1, 1, SRTargets::quadratic, 1).padded_N() == 2 * C);
    assert(make_sr_dataset_1d(C - 1, -1, 1, SRTargets::quadratic, 1).padded_N() == C);
    assert(make_sr_dataset_1d(1,     -1, 1, SRTargets::quadratic, 1).padded_N() == C);
}
void test_dataset_determinism() {
    Dataset a = make_sr_dataset_1d(50, -1, 1, SRTargets::quadratic, 42);
    Dataset b = make_sr_dataset_1d(50, -1, 1, SRTargets::quadratic, 42);
    for (size_t i = 0; i < a.inputs.size(); ++i) {
        assert(a.inputs[i]  == b.inputs[i]);
        assert(a.targets[i] == b.targets[i]);
    }
}

void test_dataset_inputs_in_range() {
    Dataset d = make_sr_dataset_1d(200, -2.5f, 2.5f, SRTargets::quadratic, 7);
    for (int n = 0; n < d.N; ++n) {
        assert(d.inputs[n] >= -2.5f);
        assert(d.inputs[n] <= 2.5f);
    }
}

// =============================================================================
// Engine integration tests
// =============================================================================
void test_engine_evaluate_all_sr_populates_fitness() {
    LGPEngine engine;
    engine.init_population();
    Dataset d = make_sr_dataset_1d(50, -1, 1, SRTargets::quadratic, 42);

    engine.evaluate_all_sr(d);

    const auto& fit = engine.get_data().fitness_scores_buf[0];
    for (auto f : fit) {
        
        assert(!std::isnan(f));
        assert(f >= 0.0f);
        assert(f <= 1.0f);
    }
}

void test_engine_view_program_slicing() {
    LGPEngine engine;
    engine.init_population();
    const auto& data = engine.get_data();

    for (int i = 0; i < LGPConfig::POPULATION_SIZE; ++i) {
        ProgramView v = engine.view_program(i);
        assert(v.instructions ==
               data.instructions_buf[0].data() + i * LGPConfig::MAX_PROGRAM_SIZE);
        assert(v.length == data.program_lengths_buf[0][i]);
    }
}
void test_tournament_returns_valid_index() {
    LGPEngine engine;
    engine.init_population();
    Dataset d = make_sr_dataset_1d(50, -1, 1, SRTargets::quadratic, 42);
    engine.evaluate_all_sr(d);

    for (int trial = 0; trial < 100; ++trial) {
        int idx = engine.tournament_selection();
        assert(idx >= 0);
        assert(idx < LGPConfig::POPULATION_SIZE);
    }
}
// iffy test works when pop is small...
void test_tournament_prefers_best() {
    LGPEngine engine;
    engine.init_population();
    // Inject known fitness: index 0 is best, everyone else is worst.
    auto& scores = engine.get_mutable_data().fitness_scores_buf[0];
    std::fill(scores.begin(), scores.end(), 0.0f);
    scores[0] = 1.0f;

    int wins = 0;
    for (int i = 0; i < 1000; ++i) {
        if (engine.tournament_selection() == 0) ++wins;
    }
    // With k=2, P(index 0 in tournament) = 1 - (POP-1/POP)^2.
    // For POP_SIZE=2: P = 0.75. For POP_SIZE=10: P = 0.19.
    // Just assert non-trivial preference — exact threshold depends on POP.
    assert(wins > 50);  // adjust for your POP_SIZE
}
// =============================================================================
// Crossover tests
//
// These assume a public engine method with the signature:
//
//   void crossover(const ProgramView& a, const ProgramView& b,
//                  uint32_t* child_a, uint8_t& child_a_len,
//                  uint32_t* child_b, uint8_t& child_b_len);
//
// The tests build parents in scratch buffers (NOT in the engine's population
// buffer) so they can control parent contents exactly. Children are written
// into MAX_PROGRAM_SIZE-sized scratch buffers, which mirrors what vary()
// will hand crossover at the real call site.
// =============================================================================

// Helper: build a parent of `len` instructions, all ADD with src1=src2=0 (so
// the encoded word's high bytes form a recognizable pattern). The dest field
// is set to `tag` so we can tell instructions from parent A apart from
// instructions from parent B in conservation checks.
static void build_tagged_parent(uint32_t* buf, int len, uint8_t tag) {
    for (int i = 0; i < len; ++i) {
        // dest = tag lets us identify which parent each instruction came from.
        // src1, src2, op are fixed; what matters is that the words are
        // distinct between the two parents.
        buf[i] = ISA::encode_manual(ISA::ADD, tag, 0, 0, false);
    }
}

void test_crossover_child_lengths() {
    // Two parents of different lengths. Child A should end up with parent B's
    // length, child B should end up with parent A's length. Run enough trials
    // to cover several different cut points.
    LGPEngine engine;
    engine.init_population();  // needed to seed RNG state; population content unused

    constexpr int LEN_A = 5;
    constexpr int LEN_B = 7;
    uint32_t parent_a_buf[LGPConfig::MAX_PROGRAM_SIZE];
    uint32_t parent_b_buf[LGPConfig::MAX_PROGRAM_SIZE];
    build_tagged_parent(parent_a_buf, LEN_A, /*tag=*/1);
    build_tagged_parent(parent_b_buf, LEN_B, /*tag=*/2);

    ProgramView pa{parent_a_buf, LEN_A};
    ProgramView pb{parent_b_buf, LEN_B};

    for (int trial = 0; trial < 100; ++trial) {
        uint32_t child_a_buf[LGPConfig::MAX_PROGRAM_SIZE] = {0};
        uint32_t child_b_buf[LGPConfig::MAX_PROGRAM_SIZE] = {0};
        uint8_t  child_a_len = 0, child_b_len = 0;

        engine.crossover(pa, pb, child_a_buf, child_a_len, child_b_buf, child_b_len);

        // Child A = a's prefix + b's suffix -> length matches parent B.
        assert(child_a_len == LEN_B);
        // Child B = b's prefix + a's suffix -> length matches parent A.
        assert(child_b_len == LEN_A);
    }
}

void test_crossover_both_parents_represented() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (default-decode semantics; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // After crossover, each child should contain instructions from BOTH parents
    // (unless the degenerate min_len < 2 path was taken -- not the case here).
    // Tag-encoded dest field makes this checkable: count how many instructions
    // in each child have dest=1 (from parent A) vs dest=2 (from parent B).
    LGPEngine engine;
    engine.init_population();

    constexpr int LEN_A = 5;
    constexpr int LEN_B = 5;
    uint32_t parent_a_buf[LGPConfig::MAX_PROGRAM_SIZE];
    uint32_t parent_b_buf[LGPConfig::MAX_PROGRAM_SIZE];
    build_tagged_parent(parent_a_buf, LEN_A, 1);
    build_tagged_parent(parent_b_buf, LEN_B, 2);

    ProgramView pa{parent_a_buf, LEN_A};
    ProgramView pb{parent_b_buf, LEN_B};

    for (int trial = 0; trial < 100; ++trial) {
        uint32_t child_a_buf[LGPConfig::MAX_PROGRAM_SIZE] = {0};
        uint32_t child_b_buf[LGPConfig::MAX_PROGRAM_SIZE] = {0};
        uint8_t  child_a_len = 0, child_b_len = 0;

        engine.crossover(pa, pb, child_a_buf, child_a_len, child_b_buf, child_b_len);

        // Count parent-of-origin in each child.
        int a_from_pa = 0, a_from_pb = 0;
        int b_from_pa = 0, b_from_pb = 0;
        for (int i = 0; i < child_a_len; ++i) {
            uint8_t d = ISA::get_dest_index(child_a_buf[i]);
            if (d == 1) ++a_from_pa;
            if (d == 2) ++a_from_pb;
        }
        for (int i = 0; i < child_b_len; ++i) {
            uint8_t d = ISA::get_dest_index(child_b_buf[i]);
            if (d == 1) ++b_from_pa;
            if (d == 2) ++b_from_pb;
        }

        // Each child must contain at least one instruction from each parent.
        // The cut point is in [1, min_len-1], guaranteeing this.
        assert(a_from_pa >= 1 && a_from_pb >= 1);
        assert(b_from_pa >= 1 && b_from_pb >= 1);

        // No spurious tags from the zero-init of the child buffer.
        assert(a_from_pa + a_from_pb == child_a_len);
        assert(b_from_pa + b_from_pb == child_b_len);
    }
}

void test_crossover_conservation() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // The multiset of instructions in (child_a ++ child_b) must equal the
    // multiset in (parent_a ++ parent_b). Single-point crossover doesn't
    // create or destroy instructions, only relocates them.
    //
    // Implementation: sort both multisets and compare element-by-element.
    // Works for any parent contents, including random ones -- so use random
    // parents to broaden coverage.
    LGPEngine engine;
    engine.init_population();

    constexpr int LEN_A = 6;
    constexpr int LEN_B = 8;
    uint32_t parent_a_buf[LGPConfig::MAX_PROGRAM_SIZE];
    uint32_t parent_b_buf[LGPConfig::MAX_PROGRAM_SIZE];

    // Hand-pick distinct instruction words so we'd notice if the same word
    // appeared more or fewer times after crossover.
    for (int i = 0; i < LEN_A; ++i) {
        parent_a_buf[i] = ISA::encode_manual(ISA::ADD, i % 8, 0, 0, false);
    }
    for (int i = 0; i < LEN_B; ++i) {
        parent_b_buf[i] = ISA::encode_manual(ISA::SUB, i % 8, 0, 0, false);
    }

    ProgramView pa{parent_a_buf, LEN_A};
    ProgramView pb{parent_b_buf, LEN_B};

    for (int trial = 0; trial < 100; ++trial) {
        uint32_t child_a_buf[LGPConfig::MAX_PROGRAM_SIZE] = {0};
        uint32_t child_b_buf[LGPConfig::MAX_PROGRAM_SIZE] = {0};
        uint8_t  child_a_len = 0, child_b_len = 0;

        engine.crossover(pa, pb, child_a_buf, child_a_len, child_b_buf, child_b_len);

        // Combined multisets must match.
        std::vector<uint32_t> parents_combined;
        parents_combined.reserve(LEN_A + LEN_B);
        for (int i = 0; i < LEN_A; ++i) parents_combined.push_back(parent_a_buf[i]);
        for (int i = 0; i < LEN_B; ++i) parents_combined.push_back(parent_b_buf[i]);

        std::vector<uint32_t> children_combined;
        children_combined.reserve(child_a_len + child_b_len);
        for (int i = 0; i < child_a_len; ++i) children_combined.push_back(child_a_buf[i]);
        for (int i = 0; i < child_b_len; ++i) children_combined.push_back(child_b_buf[i]);

        // Lengths preserved (trivially -- child_a_len = LEN_B, child_b_len = LEN_A,
        // so total = LEN_A + LEN_B). Asserted here as a sanity check.
        assert(parents_combined.size() == children_combined.size());

        std::sort(parents_combined.begin(),  parents_combined.end());
        std::sort(children_combined.begin(), children_combined.end());

        for (size_t i = 0; i < parents_combined.size(); ++i) {
            assert(parents_combined[i] == children_combined[i]);
        }
    }
}

void test_crossover_minimum_length_parents() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // Two parents of length 2 each. The only valid cut is 1, so:
    //   child_a = [a[0], b[1]]
    //   child_b = [b[0], a[1]]
    // Verify the cut isn't 0 (which would mean child_a == parent_b entirely)
    // or 2 (which would mean child_a == parent_a entirely).
    LGPEngine engine;
    engine.init_population();

    uint32_t parent_a_buf[2] = {
        ISA::encode_manual(ISA::ADD, 1, 0, 0, false),  // a[0], dest=1
        ISA::encode_manual(ISA::ADD, 1, 0, 0, false),  // a[1], dest=1
    };
    uint32_t parent_b_buf[2] = {
        ISA::encode_manual(ISA::ADD, 2, 0, 0, false),  // b[0], dest=2
        ISA::encode_manual(ISA::ADD, 2, 0, 0, false),  // b[1], dest=2
    };

    ProgramView pa{parent_a_buf, 2};
    ProgramView pb{parent_b_buf, 2};

    for (int trial = 0; trial < 50; ++trial) {
        uint32_t child_a_buf[2] = {0};
        uint32_t child_b_buf[2] = {0};
        uint8_t  child_a_len = 0, child_b_len = 0;

        engine.crossover(pa, pb, child_a_buf, child_a_len, child_b_buf, child_b_len);

        assert(child_a_len == 2);
        assert(child_b_len == 2);

        // Cut must be 1 -- so child_a[0] comes from parent A (dest=1) and
        // child_a[1] comes from parent B (dest=2). Symmetrically for child_b.
        assert(ISA::get_dest_index(child_a_buf[0]) == 1);
        assert(ISA::get_dest_index(child_a_buf[1]) == 2);
        assert(ISA::get_dest_index(child_b_buf[0]) == 2);
        assert(ISA::get_dest_index(child_b_buf[1]) == 1);
    }
}

void test_crossover_degenerate_length_one() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // If min(len_a, len_b) < 2 there's no valid cut. Spec: clone parents
    // verbatim into children. Verify that path.
    LGPEngine engine;
    engine.init_population();

    uint32_t parent_a_buf[1] = {
        ISA::encode_manual(ISA::ADD, 1, 0, 0, false),
    };
    uint32_t parent_b_buf[3] = {
        ISA::encode_manual(ISA::SUB, 2, 0, 0, false),
        ISA::encode_manual(ISA::SUB, 2, 0, 0, false),
        ISA::encode_manual(ISA::SUB, 2, 0, 0, false),
    };

    ProgramView pa{parent_a_buf, 1};
    ProgramView pb{parent_b_buf, 3};

    uint32_t child_a_buf[LGPConfig::MAX_PROGRAM_SIZE] = {0};
    uint32_t child_b_buf[LGPConfig::MAX_PROGRAM_SIZE] = {0};
    uint8_t  child_a_len = 0, child_b_len = 0;

    engine.crossover(pa, pb, child_a_buf, child_a_len, child_b_buf, child_b_len);

    // Children should be exact clones of their respective parents.
    assert(child_a_len == 1);
    assert(child_b_len == 3);
    assert(child_a_buf[0] == parent_a_buf[0]);
    for (int i = 0; i < 3; ++i) {
        assert(child_b_buf[i] == parent_b_buf[i]);
    }
}
// =============================================================================
// Mutation tests
//
// Cover both micro_mutate_instruction (single-instruction bit surgery) and
// mutate (the per-program operator that gates micro/insert/delete).
//
// The hardest bugs to catch by inspection are mask-vs-field-width issues:
// a single wrong shift bleeds bits into an adjacent field and silently
// produces "mutations" that change more than they should. test_micro_*
// targets exactly that class of bug.
// =============================================================================

void test_micro_mutate_only_one_field_changes() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // Mutate the same instruction 200 times and verify that on each call,
    // at most one of {op, dest, src1, src2_idx, src2_mode} differs from
    // the original. "At most" because the RNG can pick a new value equal
    // to the old one -- that's zero fields changed, which is also fine.
    //
    // If a mask-vs-shift bug causes one mutation to touch two fields at
    // once, this trips immediately.
    LGPEngine engine;
    engine.init_population();

    const uint32_t original = ISA::encode_manual(ISA::MUL, 3, 0, 5, true);

    for (int trial = 0; trial < 200; ++trial) {
        const uint32_t mutated = engine.micro_mutate_instruction(original);

        int diff_count = 0;
        if (ISA::get_op(original)         != ISA::get_op(mutated))         ++diff_count;
        if (ISA::get_dest_index(original) != ISA::get_dest_index(mutated)) ++diff_count;
        if (ISA::get_src1_index(original) != ISA::get_src1_index(mutated)) ++diff_count;
        if (ISA::get_src2_index(original) != ISA::get_src2_index(mutated)) ++diff_count;
        if (ISA::is_src2_constant(original) != ISA::is_src2_constant(mutated)) ++diff_count;

        assert(diff_count <= 1);
    }
}

void test_micro_mutate_decoded_fields_valid() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // Every output of micro_mutate must decode to fields within their valid
    // ranges. Catches "wrote garbage into a field" -- e.g. an unmasked
    // random shifted into position bleeding into a wider region than the
    // field actually occupies.
    LGPEngine engine;
    engine.init_population();

    const uint32_t original = ISA::encode_manual(ISA::ADD, 0, 0, 0, false);

    for (int trial = 0; trial < 500; ++trial) {
        const uint32_t mutated = engine.micro_mutate_instruction(original);

        assert(ISA::get_op(mutated)         < LGPConfig::NUM_OPERATIONS);
        assert(ISA::get_dest_index(mutated) < LGPConfig::NUM_REGISTERS);
        assert(ISA::get_src1_index(mutated) < LGPConfig::NUM_REGISTERS);
        assert(ISA::get_src2_index(mutated) < LGPConfig::NUM_REGISTERS);
        // is_src2_constant is a bool by construction; no range check needed.
    }
}

void test_micro_mutate_touches_all_fields() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // Over many trials, every field should change at least once. Catches
    // "case N is unreachable" -- e.g. a wrong dist_field range or a
    // copy-paste bug where one switch arm shadows another.
    LGPEngine engine;
    engine.init_population();

    // Pick an "all-zero in the fields we care about" base so any change is
    // visible. ADD with dest=src1=src2=0, src2 register mode.
    const uint32_t original = ISA::encode_manual(ISA::ADD, 0, 0, 0, false);

    bool op_touched = false;
    bool dest_touched = false;
    bool src1_touched = false;
    bool src2_idx_touched = false;
    bool mode_touched = false;

    for (int trial = 0; trial < 2000; ++trial) {
        const uint32_t mutated = engine.micro_mutate_instruction(original);
        if (ISA::get_op(mutated)         != ISA::ADD)  op_touched       = true;
        if (ISA::get_dest_index(mutated) != 0)         dest_touched     = true;
        if (ISA::get_src1_index(mutated) != 0)         src1_touched     = true;
        if (ISA::get_src2_index(mutated) != 0)         src2_idx_touched = true;
        if (ISA::is_src2_constant(mutated) != false)   mode_touched     = true;
    }

    assert(op_touched);
    assert(dest_touched);
    assert(src1_touched);
    assert(src2_idx_touched);
    assert(mode_touched);
}

void test_micro_mutate_preserves_unrelated_fields() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // Stronger version of "only one changes": when exactly one field
    // changes, every other field must hold its EXACT original value.
    // The "only one differs" test allows a mutation that touches a field
    // and happens to mutate it to its existing value (silent no-op) --
    // this test instead checks: for every output where field X changed,
    // fields Y, Z, W, V are bit-identical to original.
    //
    // Independently strong because mask bleed could move e.g. one bit of
    // src1 into dest while ALSO changing src1 -- "only one different" by
    // get_*-decoding accident, but the underlying bytes show two changes.
    LGPEngine engine;
    engine.init_population();

    const uint32_t original = ISA::encode_manual(ISA::DIV, 4, 7, 2, true);

    for (int trial = 0; trial < 500; ++trial) {
        const uint32_t mutated = engine.micro_mutate_instruction(original);

        // XOR isolates the bits that changed.
        const uint32_t delta = original ^ mutated;

        // Decompose delta by field. Each field's bits in delta should be
        // either entirely 0 (unchanged) or non-zero (changed). Across
        // ALL fields, at most one should be non-zero.
        const uint32_t op_bits     = (LGPConfig::OPERATION_MASK << ISA::OP_SHIFT);
        const uint32_t dest_bits   = (LGPConfig::REGISTER_MASK  << ISA::DEST_SHIFT);
        const uint32_t src1_bits   = (LGPConfig::REGISTER_MASK  << ISA::SRC1_SHIFT);
        const uint32_t src2_idx_bits = (ISA::SRC2_INDEX_MASK    << ISA::SRC2_SHIFT);
        const uint32_t mode_bits   =
            (static_cast<uint32_t>(ISA::MASK_MODE_BIT) << ISA::SRC2_SHIFT);

        int touched_field_count =
              ((delta & op_bits)       != 0 ? 1 : 0)
            + ((delta & dest_bits)     != 0 ? 1 : 0)
            + ((delta & src1_bits)     != 0 ? 1 : 0)
            + ((delta & src2_idx_bits) != 0 ? 1 : 0)
            + ((delta & mode_bits)     != 0 ? 1 : 0);

        assert(touched_field_count <= 1);

        // Bonus: delta should have NO bits outside the union of meaningful
        // fields. If it does, mutation is corrupting reserved/unused bits.
        const uint32_t meaningful =
            op_bits | dest_bits | src1_bits | src2_idx_bits | mode_bits;
        assert((delta & ~meaningful) == 0);
    }
}

// -----------------------------------------------------------------------------
// mutate() (the per-program operator)
// -----------------------------------------------------------------------------
//
// These assume the signature:
//   void LGPEngine::mutate(uint32_t* program, uint8_t& length);
//
// and the three rate constants:
//   LGPConfig::MICRO_RATE, INSERT_RATE, DELETE_RATE.

void test_mutate_zero_length_is_noop() {
    // Length-0 program must not crash and must remain length 0.
    LGPEngine engine;
    engine.init_population();

    uint32_t buf[LGPConfig::MAX_PROGRAM_SIZE] = {0};
    uint8_t  len = 0;

    for (int trial = 0; trial < 50; ++trial) {
        engine.mutate(buf, len);
        assert(len == 0);
    }
}

void test_mutate_length_stays_in_bounds() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // Run mutate many times and verify length never escapes [1, MAX].
    // Insert should refuse to grow past MAX; delete should refuse to
    // shrink below 1. If either guard is missing, this test eventually
    // trips.
    LGPEngine engine;
    engine.init_population();

    uint32_t buf[LGPConfig::MAX_PROGRAM_SIZE];
    for (int i = 0; i < LGPConfig::MAX_PROGRAM_SIZE; ++i) {
        buf[i] = ISA::encode_manual(ISA::ADD, 0, 0, 0, false);
    }
    uint8_t len = LGPConfig::STARTING_PROGRAM_SIZE;

    for (int trial = 0; trial < 5000; ++trial) {
        engine.mutate(buf, len);
        assert(len >= 1);
        assert(len <= LGPConfig::MAX_PROGRAM_SIZE);
    }
}

void test_mutate_can_grow_and_shrink() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // Over many trials, length should both grow AND shrink at least once.
    // If insert is silently broken, length only decreases (or stays put).
    // If delete is silently broken, length only increases.
    //
    // Start at a mid-range length so both growth and shrinkage have room.
    LGPEngine engine;
    engine.init_population();

    constexpr int START = LGPConfig::MAX_PROGRAM_SIZE / 2;
    uint32_t buf[LGPConfig::MAX_PROGRAM_SIZE];
    for (int i = 0; i < LGPConfig::MAX_PROGRAM_SIZE; ++i) {
        buf[i] = ISA::encode_manual(ISA::ADD, 0, 0, 0, false);
    }
    uint8_t len = START;

    bool saw_grow = false;
    bool saw_shrink = false;
    for (int trial = 0; trial < 5000 && !(saw_grow && saw_shrink); ++trial) {
        const uint8_t prev = len;
        engine.mutate(buf, len);
        if (len > prev) saw_grow = true;
        if (len < prev) saw_shrink = true;
    }

    assert(saw_grow);
    assert(saw_shrink);
}

void test_mutate_active_range_decodes_valid() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // After many mutations, every instruction in the ACTIVE range
    // [0, length) must still decode to valid fields. Catches mutation
    // writing through a stale pointer or corrupting an unrelated slot.
    LGPEngine engine;
    engine.init_population();

    uint32_t buf[LGPConfig::MAX_PROGRAM_SIZE];
    for (int i = 0; i < LGPConfig::MAX_PROGRAM_SIZE; ++i) {
        buf[i] = ISA::encode_manual(ISA::ADD, 0, 0, 0, false);
    }
    uint8_t len = LGPConfig::STARTING_PROGRAM_SIZE;

    for (int trial = 0; trial < 1000; ++trial) {
        engine.mutate(buf, len);
        for (int i = 0; i < len; ++i) {
            assert(ISA::get_op(buf[i])         < LGPConfig::NUM_OPERATIONS);
            assert(ISA::get_dest_index(buf[i]) < LGPConfig::NUM_REGISTERS);
            assert(ISA::get_src1_index(buf[i]) < LGPConfig::NUM_REGISTERS);
            assert(ISA::get_src2_index(buf[i]) < LGPConfig::NUM_REGISTERS);
        }
    }
}

void test_mutate_at_max_length_no_overflow() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // Force the program to MAX_PROGRAM_SIZE. Run mutate many times. Length
    // must never exceed MAX. Specifically targets the insert-at-tail guard.
    LGPEngine engine;
    engine.init_population();

    uint32_t buf[LGPConfig::MAX_PROGRAM_SIZE];
    for (int i = 0; i < LGPConfig::MAX_PROGRAM_SIZE; ++i) {
        buf[i] = ISA::encode_manual(ISA::ADD, 0, 0, 0, false);
    }
    uint8_t len = LGPConfig::MAX_PROGRAM_SIZE;

    for (int trial = 0; trial < 2000; ++trial) {
        engine.mutate(buf, len);
        assert(len <= LGPConfig::MAX_PROGRAM_SIZE);
    }
}

void test_mutate_at_min_length_no_underflow() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        printf("[ SKIP ] %s (hand-encoded default-decode program; INPUT_PROTECT build)\n", __func__);
        return;
    }
    // Force the program to length 1. Run mutate many times. Length must
    // never drop to 0. Targets the delete-at-tail guard.
    LGPEngine engine;
    engine.init_population();

    uint32_t buf[LGPConfig::MAX_PROGRAM_SIZE] = {
        ISA::encode_manual(ISA::ADD, 0, 0, 0, false)
    };
    uint8_t len = 1;

    for (int trial = 0; trial < 2000; ++trial) {
        engine.mutate(buf, len);
        assert(len >= 1);
    }
}
// =============================================================================
// vary_pair / vary tests
//
// These exercise the full variation pipeline. They need access to internals
// (current_buffer, the next-gen buffers) that aren't otherwise exposed, so a
// few of them lean on get_data() / get_mutable_data() and read the buffers
// directly via the [2] arrays.
//
// IMPORTANT: vary_pair and vary write into the NEXT-gen buffer. Before vary()
// flips, "next" = data.*_buf[1 - current_buffer]. After vary() flips, the old
// next-gen becomes current. Tests must be careful about which buffer they read
// at which point.
//
// Several tests need a populated, evaluated current generation as a
// precondition (vary_pair calls tournament_selection, which asserts on NaN
// fitness). The setup helper below handles that.
// =============================================================================

// Run the standard "ready to vary" setup: init + evaluate so the current
// buffer has valid fitness and tournament_selection won't trip its NaN assert.
static void setup_evaluated_engine(LGPEngine& engine, const Dataset& d) {
    engine.init_population();
    engine.evaluate_all_sr(d);
}


// -----------------------------------------------------------------------------
// vary_pair
// -----------------------------------------------------------------------------

void test_vary_pair_writes_valid_children() {
    // After vary_pair, both destination slots in next-gen must:
    //   - have length in [1, MAX]
    //   - decode to valid fields across the active range
    //   - have NaN fitness (marked unevaluated)
    Dataset d = make_sr_dataset_1d(50, -1, 1, SRTargets::quadratic, 42);
    LGPEngine engine;
    setup_evaluated_engine(engine, d);

    const PopulationData& data = engine.get_data();
    const int cur = engine.current_buffer_index();
    const int nxt = 1 - cur;

    // Use destination slots 0 and 1 in next-gen.
    engine.vary_pair(0, 1);

    for (int slot : {0, 1}) {
        const uint8_t len = data.program_lengths_buf[nxt][slot];
        assert(len >= 1);
        assert(len <= LGPConfig::MAX_PROGRAM_SIZE);

        const uint32_t* prog =
            data.instructions_buf[nxt].data() + slot * LGPConfig::MAX_PROGRAM_SIZE;
        for (int i = 0; i < len; ++i) {
            assert(ISA::get_op(prog[i])         < LGPConfig::NUM_OPERATIONS);
            assert(ISA::get_dest_index(prog[i]) < LGPConfig::NUM_REGISTERS);
            assert(ISA::get_src1_index(prog[i]) < LGPConfig::NUM_REGISTERS);
            assert(ISA::get_src2_index(prog[i]) < LGPConfig::NUM_REGISTERS);
        }

        // Children must be marked unevaluated.
        assert(std::isnan(data.fitness_scores_buf[nxt][slot]));
    }
}

void test_vary_pair_both_children_marked_nan() {
    // Specifically targets the "both lines write dstA" typo class: BOTH
    // destination fitness slots must be NaN, not just the first. If only
    // dstA were invalidated, dstB would carry a stale (non-NaN) value.
    Dataset d = make_sr_dataset_1d(50, -1, 1, SRTargets::quadratic, 42);
    LGPEngine engine;
    setup_evaluated_engine(engine, d);

    PopulationData& data = engine.get_mutable_data();
    const int cur = engine.current_buffer_index();
    const int nxt = 1 - cur;

    // Pre-poison both next-gen fitness slots with a finite value so we can
    // tell whether vary_pair actually overwrites them with NaN.
    data.fitness_scores_buf[nxt][0] = 123.0f;
    data.fitness_scores_buf[nxt][1] = 456.0f;

    engine.vary_pair(0, 1);

    assert(std::isnan(data.fitness_scores_buf[nxt][0]));
    assert(std::isnan(data.fitness_scores_buf[nxt][1]));  // the load-bearing one
}

void test_vary_pair_does_not_touch_current_buffer() {
    // vary_pair reads parents from current and writes children to next.
    // It must not modify the current buffer. Snapshot current instructions
    // and lengths before, compare after.
    Dataset d = make_sr_dataset_1d(50, -1, 1, SRTargets::quadratic, 42);
    LGPEngine engine;
    setup_evaluated_engine(engine, d);

    PopulationData& data = engine.get_mutable_data();
    const int cur = engine.current_buffer_index();

    std::vector<uint32_t> cur_instr_before = data.instructions_buf[cur];
    std::vector<uint8_t>  cur_len_before   = data.program_lengths_buf[cur];

    engine.vary_pair(0, 1);

    assert(data.instructions_buf[cur]   == cur_instr_before);
    assert(data.program_lengths_buf[cur] == cur_len_before);
}

// -----------------------------------------------------------------------------
// vary
// -----------------------------------------------------------------------------

void test_vary_flips_buffer() {
    Dataset d = make_sr_dataset_1d(50, -1, 1, SRTargets::quadratic, 42);
    LGPEngine engine;
    setup_evaluated_engine(engine, d);

    const int before = engine.current_buffer_index();
    engine.vary();
    assert(engine.current_buffer_index() != before);
}

void test_vary_produces_evaluable_population() {
    // The integration smoke test. After a full generation cycle, the new
    // population must be fully evaluable and selectable without tripping the
    // NaN assert in tournament_selection.
    Dataset d = make_sr_dataset_1d(50, -1, 1, SRTargets::quadratic, 42);
    LGPEngine engine;
    setup_evaluated_engine(engine, d);

    engine.vary();
    engine.evaluate_all_sr(d);

    // Every fitness must now be finite and in [0, 1].
    const auto& fit = engine.get_data().fitness_scores_buf[
        engine.current_buffer_index()];
    for (float f : fit) {
        assert(!std::isnan(f));
        assert(f >= 0.0f);
        assert(f <= 1.0f);
    }

    // Tournament must run cleanly many times.
    for (int i = 0; i < 100; ++i) {
        int idx = engine.tournament_selection();
        assert(idx >= 0 && idx < LGPConfig::POPULATION_SIZE);
    }
}

void test_vary_all_slots_valid_after_generation() {
    // After vary() + evaluate, every program (elite or child) must decode to
    // valid fields across its active range, with length in [1, MAX].
    Dataset d = make_sr_dataset_1d(50, -1, 1, SRTargets::quadratic, 42);
    LGPEngine engine;
    setup_evaluated_engine(engine, d);

    engine.vary();
    engine.evaluate_all_sr(d);

    const PopulationData& data = engine.get_data();
    const int cur = engine.current_buffer_index();

    for (int p = 0; p < LGPConfig::POPULATION_SIZE; ++p) {
        const uint8_t len = data.program_lengths_buf[cur][p];
        assert(len >= 1);
        assert(len <= LGPConfig::MAX_PROGRAM_SIZE);

        const uint32_t* prog =
            data.instructions_buf[cur].data() + p * LGPConfig::MAX_PROGRAM_SIZE;
        for (int i = 0; i < len; ++i) {
            assert(ISA::get_op(prog[i])         < LGPConfig::NUM_OPERATIONS);
            assert(ISA::get_dest_index(prog[i]) < LGPConfig::NUM_REGISTERS);
            assert(ISA::get_src1_index(prog[i]) < LGPConfig::NUM_REGISTERS);
            assert(ISA::get_src2_index(prog[i]) < LGPConfig::NUM_REGISTERS);
        }
    }
}

void test_vary_multi_generation_stability() {
    // Run many full generation cycles. The population must remain valid and
    // evaluable the whole way -- catches bugs that only surface after several
    // buffer flips (e.g. an accessor that doesn't actually swap, or stale-tail
    // corruption that compounds over generations).
    Dataset d = make_sr_dataset_1d(50, -1, 1, SRTargets::quadratic, 42);
    LGPEngine engine;
    setup_evaluated_engine(engine, d);

    for (int gen = 0; gen < 50; ++gen) {
        engine.vary();
        engine.evaluate_all_sr(d);

        const PopulationData& data = engine.get_data();
        const int cur = engine.current_buffer_index();
        for (int p = 0; p < LGPConfig::POPULATION_SIZE; ++p) {
            const uint8_t len = data.program_lengths_buf[cur][p];
            assert(len >= 1 && len <= LGPConfig::MAX_PROGRAM_SIZE);
            assert(!std::isnan(data.fitness_scores_buf[cur][p]));
        }
    }
}

void test_vary_best_fitness_non_decreasing_with_elitism() {
    // With elitism (ELITE_COUNT > 0), the best fitness in the population must
    // never DROP from one generation to the next -- the best programs are
    // carried forward verbatim, so the new best is at least as good.
    //
    // Skips itself if ELITE_COUNT == 0 (e.g. at POPULATION_SIZE = 2, where
    // 2 * 0.2 truncates to 0). Without elitism, best fitness can drop, so the
    // assertion wouldn't hold.
    if (LGPConfig::ELITE_COUNT == 0) {
        std::cout << "    (skipped: ELITE_COUNT == 0 at this POPULATION_SIZE)\n";
        return;
    }

    Dataset d = make_sr_dataset_1d(50, -1, 1, SRTargets::quadratic,42);
    LGPEngine engine;
    setup_evaluated_engine(engine, d);

    auto best_fitness = [&]() {
        const PopulationData& data = engine.get_data();
        const int cur = engine.current_buffer_index();
        float best = -1.0f;
        for (float f : data.fitness_scores_buf[cur]) {
            if (f > best) best = f;
        }
        return best;
    };

    float prev_best = best_fitness();
    for (int gen = 0; gen < 30; ++gen) {
        engine.vary();
        engine.evaluate_all_sr(d);
        const float cur_best = best_fitness();
        // Allow a tiny epsilon for float recomputation noise on the carried-
        // forward elite (its fitness is copied, not recomputed, so this should
        // actually be exact -- but epsilon guards against any surprise).
        assert(cur_best >= prev_best - 1e-6f);
        prev_best = cur_best;
    }
}
// ============================================================================
// Intron analysis (Introns.h): backward liveness on straight-line code.
// ============================================================================
void test_introns_dead_write_is_intron() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        // hand-encoded dests assume default decode; the liveness pass under
        // INPUT_PROTECT is covered by cuda_diff's bitwise packed==effective gate
        printf("[ SKIP ] %s (INPUT_PROTECT build)\n", __func__);
        return;
    }
    // r1 = r0 + 1  (r1 never feeds r0 -> intron); r0 = r0 * 2 (effective)
    uint32_t prog[2] = {
        ISA::encode_manual(ISA::ADD, 1, 0, 1, true),
        ISA::encode_manual(ISA::MUL, 0, 0, 5, true),   // const idx 5 = 2.0f
    };
    char mask[2];
    int cnt = Introns::effective_mask(prog, 2, mask);
    assert(cnt == 1 && mask[0] == 0 && mask[1] == 1);
}

void test_introns_unary_reads_only_src2() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        // hand-encoded dests assume default decode; the liveness pass under
        // INPUT_PROTECT is covered by cuda_diff's bitwise packed==effective gate
        printf("[ SKIP ] %s (INPUT_PROTECT build)\n", __func__);
        return;
    }
    // r2 = r0+r0 (read by SIN via src2 -> effective)
    // r3 = r0+r0 (only "read" via SIN's ignored src1 field -> intron)
    // r0 = sin(r2)
    uint32_t prog[3] = {
        ISA::encode_manual(ISA::ADD, 2, 0, 0, false),
        ISA::encode_manual(ISA::ADD, 3, 0, 0, false),
        ISA::encode_manual(ISA::SIN, 0, 3, 2, false),  // src1=r3 ignored, src2=r2 read
    };
    char mask[3];
    int cnt = Introns::effective_mask(prog, 3, mask);
    assert(cnt == 2 && mask[0] == 1 && mask[1] == 0 && mask[2] == 1);
}

void test_introns_dest_equals_src_chain() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        // hand-encoded dests assume default decode; the liveness pass under
        // INPUT_PROTECT is covered by cuda_diff's bitwise packed==effective gate
        printf("[ SKIP ] %s (INPUT_PROTECT build)\n", __func__);
        return;
    }
    // r1 = x*x ; r0 = r0 + r1 -- kill-then-mark keeps r0 live via src1.
    uint32_t prog[2] = {
        ISA::encode_manual(ISA::MUL, 1, 0, 0, false),
        ISA::encode_manual(ISA::ADD, 0, 0, 1, false),
    };
    char mask[2];
    int cnt = Introns::effective_mask(prog, 2, mask);
    assert(cnt == 2 && mask[0] == 1 && mask[1] == 1);
}

void test_introns_extract_preserves_order_and_output() {
    if constexpr (LGPConfig::INPUT_PROTECT) {
        // hand-encoded dests assume default decode; the liveness pass under
        // INPUT_PROTECT is covered by cuda_diff's bitwise packed==effective gate
        printf("[ SKIP ] %s (INPUT_PROTECT build)\n", __func__);
        return;
    }
    // Interleave introns with an effective chain; the extracted program must
    // produce a BITWISE-identical output on the CPU interpreter.
    uint32_t prog[5] = {
        ISA::encode_manual(ISA::MUL, 1, 0, 0, false),  // r1 = x*x        (effective)
        ISA::encode_manual(ISA::ADD, 4, 3, 6, true),   // r4 = r3 + 10    (intron)
        ISA::encode_manual(ISA::ADD, 1, 1, 0, false),  // r1 = r1 + x     (effective)
        ISA::encode_manual(ISA::SUB, 5, 5, 2, true),   // r5 = r5 - (-1)  (intron)
        ISA::encode_manual(ISA::ADD, 0, 1, 1, true),   // r0 = r1 + 1     (effective)
    };
    uint32_t eff[5];
    int cnt = Introns::extract_effective(prog, 5, eff);
    assert(cnt == 3);

    float inputs[LGPConfig::NUM_CONTEXTS], out_full[LGPConfig::NUM_CONTEXTS],
          out_eff[LGPConfig::NUM_CONTEXTS];
    for (int c = 0; c < LGPConfig::NUM_CONTEXTS; ++c) inputs[c] = 0.13f * c - 1.7f;
    Interpreter::run_stateless(prog, 5, inputs, 1, out_full);
    Interpreter::run_stateless(eff, cnt, inputs, 1, out_eff);
    assert(std::memcmp(out_full, out_eff, sizeof(out_full)) == 0);  // bitwise
}

// =============================================================================
// Entry point
// =============================================================================
int main() {
    // Engine init
    RUN(test_engine_init_invariants);
    RUN(test_engine_init_decoded_fields_valid);

    // Intron analysis
    RUN(test_introns_dead_write_is_intron);
    RUN(test_introns_unary_reads_only_src2);
    RUN(test_introns_dest_equals_src_chain);
    RUN(test_introns_extract_preserves_order_and_output);

    // Interpreter
    RUN(test_interpreter_doubling);
    RUN(test_interpreter_zero_init);
    RUN(test_interpreter_constant_access);
    RUN(test_interpreter_multiply);
    RUN(test_interpreter_subtract);
    RUN(test_interpreter_protected_div);
    RUN(test_interpreter_normal_div);
    RUN(test_interpreter_gt);
    RUN(test_interpreter_lt);
    RUN(test_interpreter_sin);
    RUN(test_interpreter_multi_instruction);
    RUN(test_interpreter_context_independence);
    RUN(test_interpreter_determinism);
    RUN(test_interpreter_zero_length);

    // Dataset
    RUN(test_dataset_basic_construction);
    RUN(test_dataset_padding_boundaries);
    RUN(test_dataset_determinism);
    RUN(test_dataset_inputs_in_range);

    // Engine integration
    RUN(test_engine_evaluate_all_sr_populates_fitness);
    RUN(test_engine_view_program_slicing);
    RUN(test_tournament_returns_valid_index);
    //RUN(test_tournament_prefers_best);
    // Crossover
    RUN(test_crossover_child_lengths);
    RUN(test_crossover_both_parents_represented);
    RUN(test_crossover_conservation);
    RUN(test_crossover_minimum_length_parents);
    RUN(test_crossover_degenerate_length_one);
    // Mutation
    RUN(test_micro_mutate_only_one_field_changes);
    RUN(test_micro_mutate_decoded_fields_valid);
    RUN(test_micro_mutate_touches_all_fields);
    RUN(test_micro_mutate_preserves_unrelated_fields);
    RUN(test_mutate_zero_length_is_noop);
    RUN(test_mutate_length_stays_in_bounds);
    RUN(test_mutate_can_grow_and_shrink);
    RUN(test_mutate_active_range_decodes_valid);
    RUN(test_mutate_at_max_length_no_overflow);
    RUN(test_mutate_at_min_length_no_underflow);
    // vary_pair
    RUN(test_vary_pair_writes_valid_children);
    RUN(test_vary_pair_both_children_marked_nan);
    RUN(test_vary_pair_does_not_touch_current_buffer);
    // vary
    RUN(test_vary_flips_buffer);
    RUN(test_vary_produces_evaluable_population);
    RUN(test_vary_all_slots_valid_after_generation);
    RUN(test_vary_multi_generation_stability);
    RUN(test_vary_best_fitness_non_decreasing_with_elitism);



    std::cout << "\n" << g_tests_passed << "/" << g_tests_run
              << " tests passed.\n";
    return 0;
}