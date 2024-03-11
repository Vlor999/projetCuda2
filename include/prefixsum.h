
#pragma once

#include "dataset.h"

struct PrefixSumBlending_CPU
{
    PrefixSumBlending_CPU() {};

    void run(Dataset& data, std::vector<float3>& img_out);

private:
    void prefixSumWeights(uint32_t n_pixels, uint32_t n_samples_per_pixel, const std::vector<float>& alpha_in, std::vector<float>& weights_out);
    void reduceColor(uint32_t n_pixels, uint32_t n_samples_per_pixel, const std::vector<float>& alpha_in, const std::vector<float>& weights_in, const std::vector<float3>& colors_in, std::vector<float3>& img_out);
};

struct PrefixSumBlending_GPU
{
    PrefixSumBlending_GPU();
    ~PrefixSumBlending_GPU();

    void setup(uint2 dimensions, uint32_t samples_per_pixel);
    void finalize();

    void run(DatasetGPU& data, float3* d_img_out);

private:
    void prefixSumWeights(uint32_t n_pixels, uint32_t n_samples_per_pixel, const float* d_alpha_in, float* d_weights_out);
    void reduceColor(uint32_t n_pixels, uint32_t n_samples_per_pixel, const float* d_alpha_in, const float* d_weights_in, const float3* d_colors_in, float3* d_img_out);

    float* _d_weights = nullptr; // example additional buffer
};