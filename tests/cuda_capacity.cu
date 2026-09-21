// =============================================================================
// cuda_capacity.cu — can this layout hold (and evaluate) a population of P?
//
// Allocates exactly the eval-time device footprint of an EVOLVED-like
// population for one layout, fills lengths/offsets on-device (no host
// population, so the probe is not host-RAM-bound), uploads the D=32 dataset,
// launches one real evaluation, and exits 0 on success. The OOM driver
// (tools/oom_probe.py) doubles P until this fails.
//
// Program content is left uninitialized on purpose: the ISA masks every field
// during decode, so ANY word is a valid instruction -- garbage programs
// evaluate safely (worst case inf/nan -> fitness 0). We are probing MEMORY,
// not results.
//
// Evolved-population shape (measured, P=1e4 x 1000 gens, nguyen1):
//   mean length ~30 instructions, effective fraction ~16.6%.
//
//   usage: ./cuda_capacity <layout: dense|packed|effective> <P> [meanlen] [efffrac]
// =============================================================================
#include "eval_kernels.cuh"
#include "Dataset.h"
#include "LGPConfig.h"
#include "cuda_errors.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

__global__ void fill_u32_hash(uint32_t* p, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {                       // splitmix-style hash -> reproducible random words
        unsigned long long z = (unsigned long long)i + 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        p[i] = (uint32_t)(z ^ (z >> 31));
    }
}
__global__ void fill_u8(uint8_t* p, long long n, uint8_t v) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}
__global__ void fill_cumL(long long* p, long long n1, long long stride) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n1) p[i] = i * stride;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <dense|packed|effective> <P>\n", argv[0]); return 2; }
    const std::string layout = argv[1];
    const long long P        = atoll(argv[2]);
    // int32 guard lifted 2026-08-27: the packed path now launches the
    // unchanged engine kernels in <=2^31 windows (see launch below), so P is
    // bounded by device memory, not by index width. Dense path keeps the guard.
    if (P > 2147483647LL && layout == "dense") {
        fprintf(stderr, "P exceeds int32 (dense path keeps single-launch)\n"); return 2;
    }
    const int meanlen        = (argc > 3) ? atoi(argv[3]) : 30;
    const double efffrac     = (argc > 4) ? atof(argv[4]) : 0.166;
    const int Darg           = (argc > 5) ? atoi(argv[5]) : 32;
    const int reps           = (argc > 6) ? atoi(argv[6]) : 1;

    // per-layout instruction words per individual
    long long words_per;
    bool packed;
    if      (layout == "dense")     { words_per = LGPConfig::MAX_PROGRAM_SIZE; packed = false; }
    else if (layout == "packed")    { words_per = meanlen;                     packed = true;  }
    else /* effective */            { words_per = (long long)(meanlen * efffrac + 0.999); packed = true; }

    // dataset -- resident like a real eval
    Dataset ds = make_sr_dataset_1d(Darg, -1.f, 1.f, Nguyen::Nguyen_1, 1234);
    float *d_in, *d_tg;
    ERR(cudaMalloc(&d_in, ds.padded_N() * sizeof(float)));
    ERR(cudaMalloc(&d_tg, ds.padded_N() * sizeof(float)));
    ERR(cudaMemcpy(d_in, ds.inputs.data(),  ds.padded_N() * sizeof(float), cudaMemcpyHostToDevice));
    ERR(cudaMemcpy(d_tg, ds.targets.data(), ds.padded_N() * sizeof(float), cudaMemcpyHostToDevice));
    cuda_upload_constants();

    // population buffers (the footprint under test)
    uint32_t* d_instr; float* d_fit;
    const long long nwords = P * words_per;
    ERR(cudaMalloc(&d_instr, (size_t)nwords * sizeof(uint32_t)));
    ERR(cudaMalloc(&d_fit,   (size_t)P * sizeof(float)));
    {   // deterministic random population: every word is a valid instruction
        const long long fb = (nwords + 255) / 256;
        // grid.x is 32-bit: chunk the fill
        for (long long off = 0; off < nwords; off += (1LL << 36)) { /* single pass; chunk guard */ }
        const unsigned maxg = 2147483647u;
        long long done = 0;
        while (done < nwords) {
            long long todo = nwords - done;
            long long blocks = (todo + 255) / 256;
            if (blocks > maxg) { blocks = maxg; todo = blocks * 256; }
            fill_u32_hash<<<(unsigned)blocks, 256>>>(d_instr + done, todo);
            done += todo;
        }
        ERR(cudaGetLastError()); ERR(cudaDeviceSynchronize());
    }

    const int blocks = (int)((P + 255) / 256) + 1;
    uint8_t*   d_len  = nullptr;
    long long* d_cumL = nullptr;
    if (!packed) {
        ERR(cudaMalloc(&d_len, (size_t)P * sizeof(uint8_t)));
        fill_u8<<<blocks, 256>>>(d_len, P, (uint8_t)meanlen);
    } else {
        ERR(cudaMalloc(&d_cumL, (size_t)(P + 1) * sizeof(long long)));
        fill_cumL<<<blocks, 256>>>(d_cumL, P + 1, words_per);
    }
    ERR(cudaGetLastError()); ERR(cudaDeviceSynchronize());

    const int wpb = 8;
    // Beyond int32 program counts, launch the UNCHANGED engine kernels in
    // <=2^31-1 windows via pointer offsets (CSR cumL values are absolute, so
    // d_instr stays base while d_cumL/d_fit shift). Probe-only change; the
    // verified kernels are untouched. Dense path keeps single-launch (it is
    // memory-bound far below int32 anyway).
    const long long CHUNK = 2000000000LL;
    auto launch = [&] {
        if (!packed) {
            const unsigned grid = (unsigned)((P + wpb - 1) / wpb);
            evaluate_kernel<<<grid, 256>>>(d_instr, d_len, d_fit, d_in, d_tg,
                (int)P, LGPConfig::MAX_PROGRAM_SIZE, ds.N, 1);
        } else {
            for (long long off = 0; off < P; off += CHUNK) {
                const long long n = std::min(CHUNK, P - off);
                const unsigned grid = (unsigned)((n + wpb - 1) / wpb);
                evaluate_kernel_packed<<<grid, 256>>>(d_instr, d_cumL + off,
                    d_fit + off, d_in, d_tg, (int)n, ds.N, 1);
            }
        }
    };

    launch();                                   // the capacity proof
    ERR(cudaGetLastError());
    ERR(cudaDeviceSynchronize());

    // timed launches (median of reps) -> headline throughput numbers
    cudaEvent_t a, b; ERR(cudaEventCreate(&a)); ERR(cudaEventCreate(&b));
    float best = 0.0f;
    std::vector<float> ts;
    for (int r = 0; r < reps; ++r) {
        ERR(cudaEventRecord(a));
        launch();
        ERR(cudaEventRecord(b));
        ERR(cudaEventSynchronize(b));
        float ms = 0.0f; ERR(cudaEventElapsedTime(&ms, a, b));
        ts.push_back(ms);
    }
    std::sort(ts.begin(), ts.end());
    const float med = ts[ts.size() / 2];
    printf("OK %s P=%lld (%lld words/indiv) D=%d reps=%d eval_ms=%.1f evals_per_s=%.3e gpops_s=%.3e\n",
           layout.c_str(), P, words_per, Darg, reps, med,
           P / (med / 1e3), (double)P * words_per * Darg / (med / 1e3));
    (void)best;
    return 0;
}
