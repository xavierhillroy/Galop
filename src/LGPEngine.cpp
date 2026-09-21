#include "LGPEngine.h"
#include "Introns.h"
#include <bit>
#if defined(_OPENMP) && defined(__GNUC__)
#include <parallel/algorithm>   // __gnu_parallel::nth_element / sort
#endif
#include <chrono>
#include <cstdlib>

// GALOP_PHASE_TIMERS=1: accumulate per-phase host times across a run and print
// once at the end of evolve_sr (stderr). Timing only -- no behavioral effect.
namespace {
struct PhaseTimers {
    double topk = 0, elites = 0, varyloop = 0, stats = 0;
    bool enabled = std::getenv("GALOP_PHASE_TIMERS") != nullptr;
    ~PhaseTimers() {
        if (enabled)
            fprintf(stderr, "PHASES: topk=%.3fs elites=%.3fs sel+vary=%.3fs stats=%.3fs\n",
                    topk, elites, varyloop, stats);
    }
} g_phases;
// GALOP_DIAG=path: per-generation search-dynamics CSV (observation only).
FILE* g_diag = nullptr;
bool  g_diag_tried = false;
double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}
#include "ISA.h"
#include "LGPConfig.h"
#include <random>
#include "Evaluator.h"
#include "Dataset.h"
#include <cassert>
#include <unordered_set>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include "GenerationStats.h"
#include <ostream>
#include "EvaluatorCpu.h"
void LGPEngine::init_evolution(){
    current_buffer = 0;
    current_generation = 0;
    init_population();
    // history 
    history.clear();
    history.reserve(LGPConfig::MAX_GENERATIONS);
}
void LGPEngine::init_population(){
    reset_buffers_to_sentinels();
    // sets all instructions for all programs
    for (long long i = 0; i < LGPConfig::TOTAL_INSTRUCTIONS; i ++){
        cur_instructions_mutable()[i] = generate_instruction();
       
    }
       // Set initial program lengths. Using std::fill for clarity vs vector::assign.
    std::fill(cur_lengths_mutable().begin(),
              cur_lengths_mutable().end(),
              static_cast<uint8_t>(LGPConfig::STARTING_PROGRAM_SIZE));

} // V easy GPU transformation 

// Generates instructions
template <class RNG>
uint32_t LGPEngine::generate_instruction_t(RNG& r){
    std::uniform_int_distribution<uint32_t> d32(0, UINT32_MAX);
    uint32_t raw_rand = d32(r); // gets random sttream of 32 bits
    return ISA::encode_from_random(raw_rand); // packs and cleans bits (lose some entropy)
}
uint32_t LGPEngine::generate_instruction(){
    return generate_instruction_t(rng);
}
// constructor- kept explicitly clean- inits current buffer to 0, sets gen to 0, sets up the seed, and distribution, verbose data init
LGPEngine::LGPEngine(uint32_t seed, std::unique_ptr<EvalBackend> eval)
    : current_generation(0), current_buffer(0), rng(seed),
      dist_32(0,UINT32_MAX), dist_pop(0,LGPConfig::POPULATION_SIZE-1),
      dist_unit(0.0f,1.0f), dist_field(0,4), data(),
      backend(eval ? std::move(eval) : std::make_unique<CpuEvaluator>())
{}



// This is program view, a method that returns the program view structure, contains length of cur prog and pointer to the start of it 
ProgramView LGPEngine::view_program(int i) const{
    // because we ping pong betweeen 2 buffers between generations - live bufer selection done in accessor 
    
    return ProgramView{
        cur_instructions().data() + (size_t)i * LGPConfig::MAX_PROGRAM_SIZE, // here we are getting the raw pointer of instr vector (start) and getting the adress of the prog we are working withh
        static_cast<int>(cur_lengths()[i]) // explicitly casting to int (widening from uint 8)
    };
}
// void LGPEngine::evaluate_all_sr(const Dataset& dataset){
//     for (int agent = 0; agent < LGPConfig::POPULATION_SIZE; ++agent){
//         if(std::isnan(cur_fitness()[agent])){
//             const ProgramView prog = view_program(agent);
//             cur_fitness_mutable()[agent] = Fitness::r2_to_fitness(Evaluator::evaluate_sr_r2(prog, dataset));
//         }
//     }
// }

// void LGPEngine::evaluate_all_sr(const Dataset& dataset){
    
//     backend->evaluate_population(
//         cur_instructions().data(),
//         cur_lengths().data(),
//         cur_fitness_mutable().data(),
//         LGPConfig::POPULATION_SIZE,  // 
//         dataset);
// }
// old version iwht no offset for elites - use NAN check in hotloop

void LGPEngine::evaluate_range(int start, int count, const Dataset& dataset){
    backend->evaluate_population(//skipping elites
        cur_instructions().data()    + (size_t)start * LGPConfig::MAX_PROGRAM_SIZE,  // stride!
        cur_lengths().data()         + start, 
        cur_fitness_mutable().data() + start, 
        count,
        dataset);
}

// ALL means entire populations
void LGPEngine::evaluate_all_sr(const Dataset& dataset){
    evaluate_range(0, LGPConfig::POPULATION_SIZE, dataset);
}
template <class RNG>
int LGPEngine::tournament_t(RNG& r){
    // select k agents, return the one with the highest fitness 
    //reads from the current buffer fitness fitness_scores
    std::uniform_int_distribution<int> dp(0, LGPConfig::POPULATION_SIZE - 1);

    int best_idx = dp(r); // get ran agent in population 
    float best_fit = cur_fitness()[best_idx]; // gets their fitness score 

    // make sure that fitness score is not NAN shouldnt be if evaluate all ran before xx 
    assert(!std::isnan(best_fit) && "Selection called before evaluate all (fitness NAN)");

    for (int agent = 1; agent < LGPConfig::TOURNAMENT_SIZE; ++agent){
        int cand = dp(r); // get ran agent in population 
        float cand_fit = cur_fitness()[cand]; // gets their fitness score 
        assert(!std::isnan(cand_fit) && "Selection called before evaluate all (fitness NAN)");

        if (cand_fit > best_fit){
            best_fit = cand_fit;
            best_idx = cand;
        }
    }
    return best_idx;
}
int LGPEngine::tournament_selection(){
    return tournament_t(rng);
}
// LGPEngine.cpp
#include <cstring>  // memcpy

template <class RNG>
void LGPEngine::crossover_t(const ProgramView& a, const ProgramView& b,
                            uint32_t* child_a, uint8_t& child_a_len,
                            uint32_t* child_b, uint8_t& child_b_len, RNG& r) {
    // Cut point in [1, min(len_a, len_b)). Both children get at least one
    // instruction from each parent. If either parent is length 1, we can't
    // do a meaningful crossover -- just clone both parents.

    // if (dist_unit(rng) > LGPConfig::CROSSOVER_RATE) {return;}
    const int min_len = std::min(a.length, b.length);
    if (min_len < 2) {
        std::memcpy(child_a, a.instructions, a.length * sizeof(uint32_t));
        std::memcpy(child_b, b.instructions, b.length * sizeof(uint32_t));
        child_a_len = static_cast<uint8_t>(a.length);
        child_b_len = static_cast<uint8_t>(b.length);
        return;
    }

    std::uniform_int_distribution<int> dist_cut(1, min_len - 1);
    const int cut = dist_cut(r);

    // Child A: a's prefix + b's suffix. Final length = b.length.
    std::memcpy(child_a, a.instructions, cut * sizeof(uint32_t));
    std::memcpy(child_a + cut, b.instructions + cut,
                (b.length - cut) * sizeof(uint32_t));
    child_a_len = static_cast<uint8_t>(b.length);

    // Child B: b's prefix + a's suffix. Final length = a.length.
    std::memcpy(child_b, b.instructions, cut * sizeof(uint32_t));
    std::memcpy(child_b + cut, a.instructions + cut,
                (a.length - cut) * sizeof(uint32_t));
    child_b_len = static_cast<uint8_t>(a.length);
}
void LGPEngine::crossover(const ProgramView& a, const ProgramView& b,
                          uint32_t* child_a, uint8_t& child_a_len,
                          uint32_t* child_b, uint8_t& child_b_len) {
    crossover_t(a, b, child_a, child_a_len, child_b, child_b_len, rng);
}
// Doesnt mirror crossover perfectly cuz probalistic natture baked into method... oops 
// this metthod could cause divergence on GPU 
template <class RNG>
void LGPEngine::mutate_t(uint32_t* program, uint8_t& length, RNG& r) {
    if (length == 0) return;
    std::uniform_real_distribution<float> du(0.0f, 1.0f);

    // EFFMUT (B&B "effective mutation"): compute the liveness mask once per
    // child and bias positions onto effective instructions. eff_at(k) returns
    // the k-th effective index; callers fall back to uniform if none exist.
    char effm[LGPConfig::MAX_PROGRAM_SIZE];
    int  neff = 0;
    if constexpr (LGPConfig::EFFMUT)
        neff = Introns::effective_mask(program, length, effm);
    auto eff_at = [&](int k) {
        int seen = 0;
        for (int i = 0; i < (int)length; ++i)
            if (effm[i] && seen++ == k) return i;
        return (int)length - 1;   // unreachable when k < neff
    };

    if constexpr (LGPConfig::MICRO_PER_INSTR > 0.0f) {
        // standard LGP semantics: each instruction mutates independently,
        // so mutation pressure scales with program length
        for (int i = 0; i < length; ++i)
            if (du(r) < LGPConfig::MICRO_PER_INSTR)
                program[i] = micro_mutate_instruction_t(program[i], r);
    } else if (du(r) < LGPConfig::MICRO_RATE) {
        // historical: at most ONE field of ONE instruction per child
        int idx;
        if (LGPConfig::EFFMUT && neff > 0 && du(r) < LGPConfig::EFFMUT_RATE) {
            std::uniform_int_distribution<int> dist_eff(0, neff - 1);
            idx = eff_at(dist_eff(r));            // mutate EFFECTIVE code
        } else {
            std::uniform_int_distribution<int> dist_inst(0, length - 1);
            idx = dist_inst(r);
        }
        program[idx] = micro_mutate_instruction_t(program[idx], r);
    }

    for (int ins = 0; ins < LGPConfig::MACRO_INSERTS; ++ins)
    if (du(r) < LGPConfig::INSERT_TAIL_RATE) {
        if (length < LGPConfig::MAX_PROGRAM_SIZE) {
            if constexpr (LGPConfig::MACRO_RANDOM_POS) {
                // insert BEFORE a random position in [0, length]; a <=200 B
                // memmove -- the shift-cost worry tail-only was built around
                // is ~ns at this L (measured in the tuning report)
                int pos;
                if (LGPConfig::EFFMUT && neff > 0 && du(r) < LGPConfig::EFFMUT_RATE) {
                    // insert immediately before an effective instruction: the
                    // new instruction's dest has a chance to feed live code
                    std::uniform_int_distribution<int> dist_eff(0, neff - 1);
                    pos = eff_at(dist_eff(r));
                } else {
                    std::uniform_int_distribution<int> dist_pos(0, length);
                    pos = dist_pos(r);
                }
                std::memmove(program + pos + 1, program + pos,
                             (size_t)(length - pos) * sizeof(uint32_t));
                program[pos] = generate_instruction_t(r);
            } else {
                program[length] = generate_instruction_t(r);   // tail (historical)
            }
            ++length;
        }
    }

    if (du(r) < LGPConfig::DELETE_TAIL_RATE) {
        if (length > 1) {
            if constexpr (LGPConfig::MACRO_RANDOM_POS) {
                std::uniform_int_distribution<int> dist_pos(0, length - 1);
                const int pos = dist_pos(r);
                std::memmove(program + pos, program + pos + 1,
                             (size_t)(length - 1 - pos) * sizeof(uint32_t));
            }
            --length;   // tail delete needs no move (historical path)
        }
    }
}
void LGPEngine::mutate(uint32_t* program, uint8_t& length) {
    mutate_t(program, length, rng);
}
template <class RNG>
uint32_t LGPEngine::micro_mutate_instruction_t(uint32_t instr, RNG& r_) {
    // Pick one of 5 fields to perturb. Uniform over fields, not bits --
    // gives op-changes and mode-flips comparable mutation pressure even
    // though they occupy different bit widths.
    std::uniform_int_distribution<int> d_field(0, 4);
    std::uniform_int_distribution<uint32_t> d32(0, UINT32_MAX);
    const int field = d_field(r_);  // [0, 4]
    const uint32_t r = d32(r_);     // source of random bits

    switch (field) {
        // general formula is we are creating a mask and shifting it into its appropriate spot on the ISA (32 bits)
        case 0: {  // op
            const uint32_t mask = LGPConfig::OPERATION_MASK << ISA::OP_SHIFT; 
            const uint32_t new_bits = (r & LGPConfig::OPERATION_MASK) << ISA::OP_SHIFT;
            return (instr & ~mask) | new_bits;
        }
        case 1: {  // dest
            const uint32_t mask = LGPConfig::REGISTER_MASK << ISA::DEST_SHIFT;
            const uint32_t new_bits = (r & LGPConfig::REGISTER_MASK) << ISA::DEST_SHIFT;
            return (instr & ~mask) | new_bits;
        }
        case 2: {  // src1
            const uint32_t mask = LGPConfig::REGISTER_MASK << ISA::SRC1_SHIFT;
            const uint32_t new_bits = (r & LGPConfig::REGISTER_MASK) << ISA::SRC1_SHIFT;
            return (instr & ~mask) | new_bits;
        }
        case 3: {  // src2 index (preserves the mode bit above it)
            const uint32_t mask = ISA::SRC2_INDEX_MASK << ISA::SRC2_SHIFT;
            const uint32_t new_bits = (r & ISA::SRC2_INDEX_MASK) << ISA::SRC2_SHIFT;
            return (instr & ~mask) | new_bits;
        }
        case 4: {  // src2 mode bit (toggle register/constant)
            return instr ^ (static_cast<uint32_t>(ISA::MASK_MODE_BIT) << ISA::SRC2_SHIFT);
        }
    }
    return instr;  // unreachable
}
uint32_t LGPEngine::micro_mutate_instruction(uint32_t instr) {
    return micro_mutate_instruction_t(instr, rng);
}
template <class RNG>
void LGPEngine::vary_children_t(int pA_idx, int pB_idx, int dstA, int dstB, RNG& r){
    ProgramView pA = view_program(pA_idx);
    ProgramView pB = view_program(pB_idx);

    // buffers chidren
    uint32_t* childA_buf = next_gen_ptr(dstA);
    uint32_t* childB_buf = next_gen_ptr(dstB);
    uint8_t& childA_length = next_lengths()[dstA];
    uint8_t& childB_length = next_lengths()[dstB];

    std::uniform_real_distribution<float> du(0.0f, 1.0f);

    // if we dont crossover
    if (!(du(r) < LGPConfig::CROSSOVER_RATE)){
    //clone parents 
        std::memcpy(childA_buf, pA.instructions, pA.length * sizeof(uint32_t) );// copy it exactly, destination, source, size in bytes
        std::memcpy(childB_buf, pB.instructions, pB.length * sizeof(uint32_t));

        childA_length = pA.length;
        childB_length = pB.length;
    }
    else{
        crossover_t(pA, pB, childA_buf, childA_length, childB_buf, childB_length, r);
    }
    // now do the mutation
    mutate_t(childA_buf, childA_length, r);
    mutate_t(childB_buf, childB_length, r);

    // set their fitness as Nan 
    next_fitness()[dstA] = std::numeric_limits<float>::quiet_NaN();
    next_fitness()[dstB] = std::numeric_limits<float>::quiet_NaN();
}
void LGPEngine::vary_pair(int dstA, int dstB){
    // legacy sequential path (tests): selection + variation from the master rng
    int pA_idx = tournament_selection();
    int pB_idx = tournament_selection();
    vary_children_t(pA_idx, pB_idx, dstA, dstB, rng);
}
std::vector<int> LGPEngine::top_k_indices(int k) const{
    const auto& fit = cur_fitness();
    const int P = LGPConfig::POPULATION_SIZE;

    // Elites are a SET (slot order is irrelevant downstream), so partial_sort's
    // O(P log k) is overkill -- nth_element is O(P). Keys pack the fitness bits
    // above the index: fitness is 1/(1+mse) >= 0, and non-negative floats are
    // order-isomorphic to their bit patterns, so uint64 compares == fitness
    // compares with index tie-break. Unique keys -> deterministic elite set.
    topk_keys_.resize(P);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < P; ++i)
        topk_keys_[i] = ((uint64_t)std::bit_cast<uint32_t>(fit[i]) << 32) | (uint32_t)i;

    // Keys are UNIQUE, so the top-k SET is value-determined -- any correct
    // selection algorithm returns the same set, which makes the parallel
    // nth_element safe; the parallel sort then canonicalizes slot order
    // (descending fitness), so the full result is deterministic for any
    // thread count. Non-OpenMP builds fall back to the serial algorithms.
#if defined(_OPENMP) && defined(__GNUC__)
    __gnu_parallel::nth_element(topk_keys_.begin(), topk_keys_.begin() + k,
                                topk_keys_.end(), std::greater<uint64_t>());
    __gnu_parallel::sort(topk_keys_.begin(), topk_keys_.begin() + k,
                         std::greater<uint64_t>());
#else
    std::nth_element(topk_keys_.begin(), topk_keys_.begin() + k, topk_keys_.end(),
                     std::greater<uint64_t>());
    std::sort(topk_keys_.begin(), topk_keys_.begin() + k, std::greater<uint64_t>());
#endif

    std::vector<int> idx(k);
    for (int e = 0; e < k; ++e) idx[e] = (int)(uint32_t)topk_keys_[e];
    return idx;
}
// WORTH CONSIDERING IN GPU PORT - this COPY Program copies dif amount based on length woould 
void LGPEngine::copy_elite_to_next(int srcIdx, int dstIdx){
    const uint32_t* source = cur_instructions().data() + (size_t)srcIdx * LGPConfig::MAX_PROGRAM_SIZE; // ptr to start of program at srcIdx  in current buffer (pointer to program at s)
    uint32_t* dest = next_gen_ptr(dstIdx); // pointer to program at destination index in next gen buffer 
    uint8_t length = cur_lengths()[srcIdx]; // lenght of program we are copying 

    // now the memcopy 
    std::memcpy(dest, source, sizeof(uint32_t) * length);
    next_lengths()[dstIdx] = length;
    next_fitness()[dstIdx] = cur_fitness()[srcIdx];
}

void LGPEngine::vary(){
    // fill elites - carries over length and fitness and all that good stuff 
    double t0 = now_s();
    std::vector<int> topK = top_k_indices(LGPConfig::ELITE_COUNT);
    double t1 = now_s(); g_phases.topk += t1 - t0;

    #pragma omp parallel for schedule(static)
    for (int e = 0; e < LGPConfig::ELITE_COUNT; ++e){ // FILLS first spots 
        copy_elite_to_next(topK[e], e);  // disjoint slots -> parallel-safe
    }
    double t2 = now_s(); g_phases.elites += t2 - t1;

    // One master draw per generation: decorrelates the per-pair streams across
    // generations while keeping them reproducible from LGPConfig::SEED alone.
    const uint64_t base_hi = dist_32(rng);
    const uint64_t base_lo = dist_32(rng);
    const uint64_t stream_base = (base_hi << 32) | base_lo;

    // ---- selection + variation, one parallel pass ----------------------------
    // Each pair seeds its own PCG32 from (generation base, pair index), draws
    // its two tournaments (read-only on cur_fitness), then varies into its two
    // child slots. Disjoint writes + per-pair streams -> bit-identical result
    // for any thread count, including builds without OpenMP (pragma ignored).
    const int n_pairs = LGPConfig::NON_ELITE_COUNT / 2;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_pairs; ++i) {
        PCG32 pair_rng(stream_base, (uint64_t)i);
        const int pA = tournament_t(pair_rng);
        const int pB = tournament_t(pair_rng);
        vary_children_t(pA, pB,
                        LGPConfig::ELITE_COUNT + 2*i, LGPConfig::ELITE_COUNT + 2*i + 1,
                        pair_rng);
    }
    g_phases.varyloop += now_s() - t2;
    flip_generation();
}
void LGPEngine::reset_buffers_to_sentinels() {
    for (int b = 0; b < 2; ++b) {
        std::fill(data.fitness_scores_buf[b].begin(),
                  data.fitness_scores_buf[b].end(),
                  std::numeric_limits<float>::quiet_NaN());
        std::fill(data.program_lengths_buf[b].begin(),
                  data.program_lengths_buf[b].end(),
                  static_cast<uint8_t>(LGPConfig::SIZE_SENTINEL));
    }
}
GenerationStats LGPEngine::compute_stats() const {
    const auto& fit = cur_fitness();
    const auto& len = cur_lengths();
    const int P = LGPConfig::POPULATION_SIZE;

    // Fixed chunk count (NOT the thread count), so the float-summation order --
    // and therefore the result -- is identical for any OMP_NUM_THREADS.
    // Per-chunk double accumulation also fixes the float-precision mush of a
    // single running sum at P >= 1e6, and len_sum is long long (the previous
    // int overflowed above P ~ 7e7).
    constexpr int C = 256;
    double    csum[C];
    long long clen[C];
    float     cbest[C];
    int       cidx[C];
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; ++c) {
        const long long lo = (long long)P * c / C, hi = (long long)P * (c + 1) / C;
        double s = 0.0; long long ls = 0; float b = -1.0f; int bi = (int)lo;
        for (long long i = lo; i < hi; ++i) {
            ls += len[i];
            s  += fit[i];
            if (fit[i] > b) { b = fit[i]; bi = (int)i; }
        }
        csum[c] = s; clen[c] = ls; cbest[c] = b; cidx[c] = bi;
    }
    double sum = 0.0; long long len_sum = 0; float best = -1.0f; int best_idx = 0;
    for (int c = 0; c < C; ++c) {          // fixed combine order; strict > keeps
        sum += csum[c]; len_sum += clen[c]; // the lowest index on fitness ties,
        if (cbest[c] > best) { best = cbest[c]; best_idx = cidx[c]; } // like the old scan
    }

    return GenerationStats{
        current_generation,
        best,
        static_cast<float>(sum / P),
        static_cast<int>(len[best_idx]),
        static_cast<float>((double)len_sum / P),
        best_idx
    };
}
void LGPEngine::evolve_sr(const Dataset& dataset){
    init_evolution(); // set programs to random instructs, resets all buff and gen counts

    evaluate_all_sr(dataset);
    history.push_back(compute_stats());

    if (!g_diag_tried) {
        g_diag_tried = true;
        if (const char* p = std::getenv("GALOP_DIAG")) {
            g_diag = fopen(p, "w");
            if (g_diag) fprintf(g_diag, "gen,best_fitness,mean_fitness,unique_fitness,eff_fraction,mean_len\n");
        }
    }
    for (int gen = 1; gen < LGPConfig::MAX_GENERATIONS; ++gen){
        current_generation = gen;
        vary(); // this does tthe flip 
        evaluate_range(LGPConfig::ELITE_COUNT,
                   LGPConfig::NON_ELITE_COUNT, dataset); // skips elites 
        { double ts = now_s(); history.push_back(compute_stats()); g_phases.stats += now_s() - ts; } // based on cur gen - if something flips then reading from wrong buffer
        if (g_diag) {
            const auto& st = history.back();
            const auto& fit = cur_fitness();
            const auto& len = cur_lengths();
            std::unordered_set<uint32_t> uniq;                 // diversity proxy
            long long sumL = 0, sumEff = 0;
            char m[LGPConfig::MAX_PROGRAM_SIZE];
            for (int i = 0; i < LGPConfig::POPULATION_SIZE; ++i) {
                uniq.insert(std::bit_cast<uint32_t>(fit[i]));
                sumL += len[i];
                sumEff += Introns::effective_mask(
                    cur_instructions().data() + (size_t)i * LGPConfig::MAX_PROGRAM_SIZE,
                    len[i], m);
            }
            fprintf(g_diag, "%d,%.7e,%.7e,%zu,%.4f,%.2f\n", gen, st.best_fitness,
                    st.mean_fitness, uniq.size(),
                    sumL ? (double)sumEff / sumL : 0.0,
                    (double)sumL / LGPConfig::POPULATION_SIZE);
            fflush(g_diag);
        }
    }
}
void LGPEngine::print_history() const {
    std::cout << "gen,best,mean,best_len\n";
    for (const auto& s : history) {
        std::cout << s.generation << ","
                  << s.best_fitness << ","
                  << s.mean_fitness << ","
                  << s.best_length << "," 
                  <<s.mean_length <<"\n";
    }
}
// Helper: opcode -> printable symbol. Lives near print_best_program (or in
// ISA.h if you want it shared). Returns infix symbol for binary ops, a name
// for unary/function ops.
static const char* op_symbol(uint8_t op) {
    switch (op) {
        case ISA::ADD: return "+";
        case ISA::SUB: return "-";
        case ISA::MUL: return "*";
        case ISA::DIV: return "/";
        case ISA::LT:  return "<";
        case ISA::GT:  return ">";
        case ISA::SIN: return "sin";
        case ISA::COS: return "cos";
        default:       return "?";
    }
}

void LGPEngine::print_best_program(std::ostream& os) const {
    const GenerationStats& last = history.back();
    const uint32_t* prog = cur_instructions().data()
                         + (size_t)last.best_index * LGPConfig::MAX_PROGRAM_SIZE;

    os << "Best program (fitness " << last.best_fitness
              << ", length " << last.best_length << "):\n";

    for (int i = 0; i < last.best_length; ++i) {
        const uint32_t instr = prog[i];
        const uint8_t  op    = ISA::get_op(instr);
        const uint8_t  dest  = ISA::get_dest_index(instr);
        const uint8_t  src1  = ISA::get_src1_index(instr);
        const uint8_t  src2  = ISA::get_src2_index(instr);
        const bool     s2c   = ISA::is_src2_constant(instr);

        os << "  [" << i << "] r" << (int)dest << " = ";

        if (op == ISA::SIN || op == ISA::COS) {
            // Unary: reads src2 (which may be a register or a constant),
            // ignores src1. Print the function form.
            os << op_symbol(op) << "(";
            if (s2c) os << LGPConfig::CONSTANTS[src2];
            else     os<< "r" << (int)src2;
            os << ")";
        } else {
            // Binary: r[src1] <op> (r[src2] or CONSTANTS[src2])
           os<< "r" << (int)src1 << " " << op_symbol(op) << " ";
            if (s2c) os << LGPConfig::CONSTANTS[src2];
            else     os << "r" << (int)src2;
        }

        os << "\n";
    }
}
// Best program's MSE on the training data it evolved against. Stored fitness is
// mse_to_fitness(mse) = 1/(1+mse); invert to recover the MSE.
float LGPEngine::best_train_mse() const {
    const float f = history.back().best_fitness;
    return f > 0.0f ? (1.0f / f - 1.0f) : std::numeric_limits<float>::infinity();
}

// Best program's MSE on a different (held-out) dataset -- the honest
// generalization signal for the test column.
float LGPEngine::best_test_mse(const Dataset& test) const {
    const int idx = history.back().best_index;
    const ProgramView prog = view_program(idx);
    return Evaluator::evaluate_sr_mse(prog, test);
}

int LGPEngine::best_length() const {
    return history.back().best_length;
}
