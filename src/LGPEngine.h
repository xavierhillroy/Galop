#ifndef LGP_ENGINE_H
#define LGP_ENGINE_H
#include "LGPConfig.h"
#include <cstdint>
#include <random>
#include <vector>
#include <limits>
#include "Evaluator.h"
#include "Dataset.h"
#include "GenerationStats.h"
#include <ostream>
#include <iostream>
#include "EvalBackend.h"
#include <memory>
#include <type_traits>

// =============================================================================
// GALOP_PARALLEL_INIT (build flag, default OFF): population initialization in
// parallel. OFF, the engine is exactly the shipped one: the instruction buffers
// are zero-filled at construction and init_population() is one serial pass of
// the engine's mt19937 over every P x MAX_PROGRAM_SIZE slot. ON, the buffers
// are allocated without that zero pass (InstrVec below) and init_population()
// generates program i from its own PCG32 stream (seed, i) in an OpenMP loop,
// so the population is a function of the seed alone, identical for any thread
// count -- but DIFFERENT from the serial initializer's population: the flag
// changes initialization time only and must never be mixed with results
// measured without it.
// =============================================================================
#ifdef GALOP_PARALLEL_INIT
template <class T>
struct DefaultInitAlloc : std::allocator<T> {       // value-initialization becomes default-initialization: no zero pass
    template <class U> struct rebind { using other = DefaultInitAlloc<U>; };
    using std::allocator<T>::allocator;
    template <class U> void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) { ::new (static_cast<void*>(p)) U; }
    template <class U, class... A> void construct(U* p, A&&... a) { ::new (static_cast<void*>(p)) U(std::forward<A>(a)...); }
};
using InstrVec = std::vector<uint32_t, DefaultInitAlloc<uint32_t>>;
#else
using InstrVec = std::vector<uint32_t>;
#endif

// =============================================================================
// PopulationData: owns all the buffers that describe the population.
//
// This struct is deliberately just data + allocation -- it doesn't implement
// any of the evolutionary loop. LGPEngine operates on it. Keeping the two
// separate means (a) PopulationData can later be swapped for a GPU-backed
// equivalent (same fields, but storage is cudaMalloc'd) without touching
// engine logic, and (b) tests can construct a PopulationData in isolation.
//
// Memory layout recap (see LGPConfig.h for the "why"):
//   - instructions[2][]:    flat buffer of TOTAL_INSTRUCTIONS uint32_t words.
//                        Program i occupies the slice
//                          [i * MAX_PROGRAM_SIZE, i * MAX_PROGRAM_SIZE + program_lengths[i])
//                        Slots past program_lengths[i] exist but aren't
//                        executed.- 
//   - program_lengths[]: one byte per program; gates how far the interpreter
//                        walks into that program's slice.
//
// three parallel "next-gen" buffers implement ping-pong double-buffering for
// variation: children are written into next_gen_* while the current
// generation remains readable. After variation completes, the engine flips
// a pointer/flag  to swap roles rather than copying. This
// avoids allocation churn per generation and mirrors how the GPU version
// will want to work.
// =============================================================================

// =============================================================================
// PCG32: per-pair RNG for the parallel variation pass (O'Neill's pcg32).
//
// Why not mt19937 per pair: mt19937 construction fills a 624-word table, which
// at P=10^8 (~5e7 pairs/generation) costs more than the variation work itself.
// PCG32 init is two multiplies. Why not a plain counter hash: each pair needs
// a *stream* of draws, and counter-line generators seeded near each other can
// land on the same orbit; pcg32's per-stream increment (2*stream+1, odd) gives
// provably distinct sequences for distinct pair indices.
//
// Satisfies UniformRandomBitGenerator, so std::uniform_*_distribution works.
// =============================================================================
struct PCG32 {
    uint64_t state;
    uint64_t inc;
    PCG32(uint64_t seed, uint64_t stream) : state(0u), inc((stream << 1u) | 1u) {
        (*this)();
        state += seed;
        (*this)();
    }
    using result_type = uint32_t;
    static constexpr uint32_t min() { return 0; }
    static constexpr uint32_t max() { return 0xFFFFFFFFu; }
    uint32_t operator()() {
        const uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        const uint32_t xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
        const uint32_t rot = static_cast<uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
    }
};

// Container that contains population information
struct PopulationData {

    // ---- Current generation ------------------------------------------------
    InstrVec instructions_buf[2];                   // Flat buffer of all programs'
                                             // instructions; size = TOTAL_INSTRUCTIONS.
    std::vector<uint8_t>  program_lengths_buf[2];   // Active length of each program.
                                             // uint8_t because MAX_PROGRAM_SIZE fits
                                             // comfortably under 255.
    static_assert(LGPConfig::MAX_PROGRAM_SIZE <= LGPConfig::SIZE_SENTINEL, "MAX PROGRAM SIZE LARGER THEN uint8 can store");
    std::vector<float>fitness_scores_buf[2];

    // -------------------------------------------------------------------------
    // Default constructor sizes all buffers from the compile-time constants
    // in LGPConfig, so the struct is always ready-to-use after construction.
    // No manual resize() calls at the use site.
    //
    // Initialization values chosen deliberately:
    //   - instructions[i] -> 0: zero-filled instruction
    //     slots decode to a valid (if meaningless) instruction, so a missed
    //     init() can't produce UB in the interpreter.
    //   - program_lengths -> STARTING_PROGRAM_SIZE: same defensive reasoning.
    //     A missed init() still leaves every program with a valid length.
    //   -
    //   - fitness_scores -> NaN: sentinel for "not yet evaluated". Any
    //     comparison with NaN returns false, so selection run on an
    //     unevaluated population fails visibly rather than silently picking
    //     the zero-fitness winners.
    // -------------------------------------------------------------------------
    PopulationData(){
        for (int b =0; b <2; ++b){
#ifdef GALOP_PARALLEL_INIT
            instructions_buf[b].resize(LGPConfig::TOTAL_INSTRUCTIONS);   // allocated, not touched: init_population() is the first touch
#else
            instructions_buf[b].assign(LGPConfig::TOTAL_INSTRUCTIONS, 0);
#endif
            program_lengths_buf[b].assign(LGPConfig::POPULATION_SIZE, LGPConfig::SIZE_SENTINEL);// invalid sentinel is 255 ( largest val we can read in uint8 )
            fitness_scores_buf[b].assign(LGPConfig::POPULATION_SIZE, std::numeric_limits<float>::quiet_NaN());
        }
    }
   
};

// =============================================================================
// LGPEngine: owns the evolutionary loop.
//
// Holds the population data, the RNG, and the generation counter, and
// exposes the operators (init / mutate / crossover / select / vary / evolve)
// that drive the loop. All randomness flows through the engine's single
// mt19937 so that runs are reproducible given LGPConfig::SEED.
// =============================================================================
class LGPEngine {
private:
    // ---- Generation state --------------------------------------------------
    int current_generation;                        // Counter; starts at 0 in ctor.
    int current_buffer;                            // 0 or 1: indicates which of
                                                   // the two buffers in PopulationData
                                                   // is the "live" generation.
                                                   // Flipped after each variation pass.

    // ---- RNG ---------------------------------------------------------------
    std::mt19937 rng;      // Seeded with LGPConfig::SEED
                                                        // in the constructor.
    std::uniform_int_distribution<uint32_t>   dist_32;  // Uniform over the full
                                                        // uint32_t range; used to
                                                        // generate random instruction
                                                        // words.
    //Population distribution, random over 0 -> popsize - 1
    std::uniform_int_distribution<int> dist_pop;

    // Members to add:
    std::uniform_real_distribution<float> dist_unit;       // [0, 1)
    std::uniform_int_distribution<int> dist_field;      // [0, 4]

    // ---- Population --------------------------------------------------------
    PopulationData data;                           // All buffers; default-constructed.
    std::vector<GenerationStats> history;
    std::unique_ptr<EvalBackend> backend; // pointer to backend we use (EIther CPU/ GPU or RL or SR )



public:
    explicit LGPEngine(uint32_t seed = LGPConfig::SEED, std::unique_ptr<EvalBackend> eval = nullptr);
    ~LGPEngine() = default;
    ProgramView view_program(int i) const; // returns a program view object ( cur instruction part and cur length)
    void evaluate_all_sr(const Dataset& dataset); // evluates entire population... this loop will disappear on GPU - be assigned to diff warps 
    void evaluate_all_rl();
    // ---- Public evolutionary interface -------------------------------------
    void init_population();                  // Randomize the initial population.
    void mutate(uint32_t* program, uint8_t& length);
    void crossover(const ProgramView& a, const ProgramView& b,
                          uint32_t* child_a, uint8_t& child_a_len,
                          uint32_t* child_b, uint8_t& child_b_len);
    int  tournament_selection();  // Returns the program index that is selected.
    void vary();                  // Crossover + mutation over the whole population.
    void evolve_sr(const Dataset& dataset); 
    
    // TO DO // Top-level evolutionary loop.
    void evolve_rl();


    const PopulationData& get_data() const { return data; } // access data for testing 
    PopulationData& get_mutable_data() { return data; } // access data for testing 
    uint32_t micro_mutate_instruction(uint32_t instruction);
    
    void vary_pair(int dstA, int dstB);
    
    int current_buffer_index() const { return current_buffer; }
    void init_evolution();
    void print_best_program(std::ostream& os = std::cout) const;
    void print_history() const;
    // void print_best_program() const;
    
    


private:
    // ---- Internal helpers --------------------------------------------------
    uint32_t generate_instruction();  // One random, encoding-valid instruction word.

    // ---- Operator implementations, generic over the RNG ---------------------
    // Single source of truth for the operators. The public mutate/crossover/
    // micro_mutate wrappers call these with the engine's mt19937 (sequential,
    // test-visible path); the parallel variation pass in vary() calls them with
    // a per-pair PCG32. Local std::uniform_*_distribution objects inside are
    // stateless for these parameterizations, so wrapper behavior is identical
    // to the historical member-distribution code.
    template <class RNG> uint32_t generate_instruction_t(RNG& r);
    template <class RNG> uint32_t micro_mutate_instruction_t(uint32_t instr, RNG& r);
    template <class RNG> void mutate_t(uint32_t* program, uint8_t& length, RNG& r);
    template <class RNG> void crossover_t(const ProgramView& a, const ProgramView& b,
                                          uint32_t* child_a, uint8_t& child_a_len,
                                          uint32_t* child_b, uint8_t& child_b_len, RNG& r);
    // Variation of one selected pair into next-gen slots dstA/dstB (no selection,
    // no shared state: writes are disjoint per pair -> safe under OpenMP).
    template <class RNG> void vary_children_t(int pA_idx, int pB_idx,
                                              int dstA, int dstB, RNG& r);
    // Tournament over cur_fitness with the given RNG (read-only on shared
    // state -> safe to run inside the parallel pair loop).
    template <class RNG> int tournament_t(RNG& r);

    mutable std::vector<uint64_t> topk_keys_;  // top_k scratch; member so the
                                               // allocation persists across generations

    // readers
    const InstrVec& cur_instructions() const{ // returns reference to current instruction buffer 
        return data.instructions_buf[current_buffer];
    } 
    const std::vector<uint8_t>& cur_lengths() const {
        return data.program_lengths_buf[current_buffer];
    }
    const std::vector<float>& cur_fitness() const{
        return data.fitness_scores_buf[current_buffer];
    }
    // write accessor - where to write next gen + Current gen fitness  - (KEY THESE GIVE YOU THE NEXT GEN EMBEDDED)
    InstrVec& next_instructions(){
        return data.instructions_buf[1 - current_buffer ]; //if buff is 0, 1 - 0 = 1 flips, if buff is 1, 1-1  = 0, so it flips 
    }
    std::vector<uint8_t>& next_lengths(){
        return data.program_lengths_buf[1 - current_buffer];
    }
    std::vector<float>& next_fitness(){
        return data.fitness_scores_buf[1 - current_buffer];
    }

    std::vector<float>& cur_fitness_mutable(){
        return data.fitness_scores_buf[current_buffer];
    }
    InstrVec& cur_instructions_mutable(){
        return data.instructions_buf[current_buffer];
    }
    std::vector<uint8_t>& cur_lengths_mutable()  {
        return data.program_lengths_buf[current_buffer];
    }
    void flip_generation(){
        current_buffer = 1 - current_buffer;
    }
    uint32_t* next_gen_ptr(int i){ // gives us the pointer to the start of program at index i of next gen ss
        return next_instructions().data() + (size_t)i * LGPConfig::MAX_PROGRAM_SIZE;
    }
    std::vector<int> top_k_indices(int k) const;

    void copy_elite_to_next(int srcIdx, int dstIdx); //given idx from source program from cur gen, copy that program to next gen at a the dstIndex - for elites... we copy fitness too!
    
    void reset_buffers_to_sentinels();
    GenerationStats compute_stats() const;
    // header (private): evaluate programs [start, start+count) via the backend
    void evaluate_range(int start, int count, const Dataset& dataset);
    public:
    float best_train_mse() const;
    float best_test_mse(const Dataset& test) const;
    int best_length() const;
    const std::vector<GenerationStats>& history_view() const { return history; }


};


#endif  // LGP_ENGINE_H