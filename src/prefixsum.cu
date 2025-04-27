
#include "prefixsum.h"
#include "helper/helper_math.h"
#include "helper/cuda_helper_host.h"

#include <cub/cub.cuh>

#define BLOCK 256 // 512 does not inprove anything 

PrefixSumBlending_GPU::PrefixSumBlending_GPU()
{
}

PrefixSumBlending_GPU::~PrefixSumBlending_GPU()
{
}

void PrefixSumBlending_GPU::setup(uint2 dimensions, uint32_t samples_per_pixel)
{
    uint32_t n_pixels = dimensions.x * dimensions.y;

    // example: allocate additional buffer (malloc not relevant for timing)
    CUDA_CHECK_THROW(cudaMalloc(&_d_weights, sizeof(float) * n_pixels * samples_per_pixel));

    CUDA_SYNC_CHECK_THROW();
}

void PrefixSumBlending_GPU::finalize()
{
    if (_d_weights)
        cudaFree(_d_weights);
}

void PrefixSumBlending_GPU::run(DatasetGPU &data, float3 *d_img_out)
{
    // Do not store and reuse calculations between different runs! You have to do the full computation each run.
    this->prefixSumWeights(data._n_pixels, data._samples_per_pixel, data._alphas, _d_weights);
    this->reduceColor(data._n_pixels, data._samples_per_pixel, data._alphas, _d_weights, data._colors, d_img_out);
}

__global__ void prefixSumWeightsIteration(const float* alpha_in, float* weights_out, uint32_t n_samples_per_pixel) {
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    const float initVal = 1.0f;

    const float* alphas = alpha_in + pos * n_samples_per_pixel;
    float* weights = weights_out + pos * n_samples_per_pixel;

    float acc = 1.0f;

    for (uint32_t i = 0; i < n_samples_per_pixel; ++i) {
        weights[i] = acc;
        acc *= (initVal - alphas[i]);
    }
}

void PrefixSumBlending_GPU::prefixSumWeights(uint32_t n_pixels, uint32_t n_samples_per_pixel, const float *d_alpha_in, float *d_weights_out)
{
    // Equation : 2
    uint32_t grid = (n_pixels + BLOCK - 1) / BLOCK;
    prefixSumWeightsIteration<<<grid, BLOCK>>>(d_alpha_in, d_weights_out, n_samples_per_pixel);

    CUDA_SYNC_CHECK_THROW();
}

__global__ void reduceColorIteration(uint32_t n_pixels, uint32_t n_samples_per_pixel, const float *alpha, const float *weights, const float3 *colors, float3 *img_out) {
    // Equation : 1
    uint32_t pos = blockIdx.x * blockDim.x + threadIdx.x;

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    uint32_t base = pos * n_samples_per_pixel;
    for (uint32_t i = 0; i < n_samples_per_pixel; ++i) {
        float ai = alpha[base + i];
        float ti = weights[base + i];
        float3 ci = colors[base + i];
        sum += ci * (ai * ti);
    }
    img_out[pos] = sum;
}

void PrefixSumBlending_GPU::reduceColor(uint32_t n_pixels, uint32_t n_samples_per_pixel, const float *d_alpha_in, const float *d_weights_in, const float3 *d_colors_in, float3 *d_img_out)
{
    // Equation : 1
    uint32_t grid = (n_pixels + BLOCK - 1) / BLOCK;
    reduceColorIteration<<<grid, BLOCK>>>(n_pixels, n_samples_per_pixel, d_alpha_in, d_weights_in, d_colors_in, d_img_out);

    CUDA_SYNC_CHECK_THROW();
}