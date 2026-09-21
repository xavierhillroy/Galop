#ifndef GENERATION_STATS_H
#define GENERATION_STATS_H

struct GenerationStats{
    int generation;
    float best_fitness;
    float mean_fitness;
    int best_length;
    float mean_length;
    int best_index;
};
#endif