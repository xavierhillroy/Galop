#include "Dataset.h"
#include <random>
#include "LGPConfig.h"
#include <cmath>
#include <fstream>
#include <iomanip>   
#include <stdexcept>

namespace SRTargets{
    // simple sr functions 
    float quadratic(float x ){return x*x + x;}
    float koza1(float x){return x*x*x*x + x*x*x +x*x +x;}
    float identity(float x) {return x;} 
    float koza3(float x) {
        const float x2 = x * x;
        const float x3 = x2 * x;
        const float x5 = x3 * x2;
        return x5 - 2.0f * x3 + x;
    }
}
namespace Nguyen{
    float Nguyen_1(float x) {return x*x*x + x*x + x;}
    float Nguyen_2(float x) {return x*x*x*x +x*x*x + x*x + x;}
    float Nguyen_3(float x) {return x*x*x*x*x + x*x*x*x +x*x*x + x*x + x;}
    float Nguyen_4(float x) {return x*x*x*x*x*x + x*x*x*x*x + x*x*x*x +x*x*x + x*x + x;}
    float Nguyen_5(float x) {return std::sin(x*x)*std::cos(x) -1;}
    float Nguyen_6(float x) {return std::sin(x) + sin(x+ x*x);}

    
}
// Compute SS_tot = Σ(y - ȳ)² over the N valid targets (excludes padding).
// Two-pass: mean first, then squared deviations. More numerically stable
// than the one-pass Σy² - (Σy)²/N form, which suffers catastrophic
// cancellation when the mean is large relative to the spread.
static double compute_ss_tot(const std::vector<float>& targets, int N) {
     if (N <= 0) return 0.0;   
    double mean = 0.0;
    for (int i = 0; i < N; ++i) mean += double(targets[i]);
    mean /= double(N);

    double ss = 0.0;
    for (int i = 0; i < N; ++i) {
        const double d = double(targets[i]) - mean;
        ss += d * d;
    }
    return ss;
}

Dataset make_sr_dataset_1d(int N, float xmin, float xmax,float (*target_fn)(float), uint32_t seed){
    Dataset d;
    d.N = N; 
    d.num_inputs = 1; // since 1d

    // allocated padded inputs - padded to context size 32 ;)
    const int padded = d.padded_N();
    d.inputs.assign(padded * d.num_inputs, 0.0f);
    d.targets.assign(padded, 0.0f);

    //setting up RNG using global seed we had for repro
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(xmin,xmax);


    // filling up  vectors up to N (extra parts already padded to 0)
    for (int n = 0; n <N; ++n ){
        const float x = dist(rng);
        d.inputs[n] = x; // stride 1 because d.inputs is 1 
        d.targets[n] = target_fn(x);
    }
    d.ss_tot = compute_ss_tot(d.targets, N);  // cache once
    return d; // return finished dataset x 
}


void write_csv(const Dataset& d, const std::string& path) {
    std::ofstream f(path);
    f << std::setprecision(9);          // float has ~7 sig digits; 9 is lossless
    f << "x,y\n";
    for (int n = 0; n < d.N; ++n)
        f << d.inputs[n] << "," << d.targets[n] << "\n";

}
Dataset load_csv_1d(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);

    std::string line;
    std::getline(f, line);  // skip header "x,y"

    std::vector<float> xs, ys;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const auto comma = line.find(',');
        if (comma == std::string::npos) continue;
        xs.push_back(std::stof(line.substr(0, comma)));
        ys.push_back(std::stof(line.substr(comma + 1)));
    }

    Dataset d;
    d.N = static_cast<int>(xs.size());
    d.num_inputs = 1;

    const int padded = d.padded_N();
    d.inputs.assign(padded * d.num_inputs, 0.0f);
    d.targets.assign(padded, 0.0f);
    for (int n = 0; n < d.N; ++n) {
        d.inputs[n] = xs[n];
        d.targets[n] = ys[n];
    }
    d.ss_tot = compute_ss_tot(d.targets, d.N);  // same helper as builder
    return d;
}

Dataset load_csv_multi(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);

    std::string line;
    std::getline(f, line);  // header row names the columns; count = features + 1

    std::vector<std::vector<float>> rows;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<float> row;
        size_t start = 0;
        while (start <= line.size()) {
            const auto comma = line.find(',', start);
            const auto end = (comma == std::string::npos) ? line.size() : comma;
            row.push_back(std::stof(line.substr(start, end - start)));
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        rows.push_back(std::move(row));
    }
    if (rows.empty() || rows[0].size() < 2)
        throw std::runtime_error("multi-feature csv needs >= 2 columns: " + path);

    Dataset d;
    d.N = static_cast<int>(rows.size());
    d.num_inputs = static_cast<int>(rows[0].size()) - 1;

    const int padded = d.padded_N();
    d.inputs.assign((size_t)padded * d.num_inputs, 0.0f);
    d.targets.assign(padded, 0.0f);
    for (int n = 0; n < d.N; ++n) {
        for (int i = 0; i < d.num_inputs; ++i)
            d.inputs[(size_t)n * d.num_inputs + i] = rows[n][i];
        d.targets[n] = rows[n][d.num_inputs];
    }
    d.ss_tot = compute_ss_tot(d.targets, d.N);
    return d;
}