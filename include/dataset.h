
#pragma once

#include <filesystem>
#include <vector>
namespace fs = std::filesystem;

#include "cuda_runtime.h"

struct DatasetGPU
{
    uint2 _dimensions; // (width (W), height (H))
    uint32_t _n_pixels; // N_P = W * H

    uint32_t _samples_per_pixel; // N_S
    float3* _colors; // size: sizeof(float) * N_P * N_S * 3
    float* _alphas; // size: sizeof(float) * N_P * N_S
};

struct Dataset
{
    Dataset() {};

    bool load(fs::path input_dir);
    void unpack();
    DatasetGPU upload();

    uint2 _dimensions; // (width (W), height (H))
    uint32_t _n_pixels; // N_P = W * H
    std::vector<uint32_t> _counts, _offsets;

    uint32_t _total_packed_count;
    std::vector<float3> _colors_packed;
    std::vector<float> _alphas_packed;

    uint32_t _samples_per_pixel; // N_S
    std::vector<float3> _colors_unpacked; // size: N_P * N_S * 3
    std::vector<float> _alphas_unpacked;  // size: N_P * N_S
};