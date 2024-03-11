
#include "prefixsum.h"
#include "helper/helper_math.h"
#include "helper/cuda_helper_host.h"

#include <cub/cub.cuh>

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

void PrefixSumBlending_GPU::prefixSumWeights(uint32_t n_pixels, uint32_t n_samples_per_pixel, const float *d_alpha_in, float *d_weights_out)
{
    /*
        Task 1:
        TODO: Compute the parallel prefix sum for the transmittance values T_i (see Equation (2) in assignment sheet)
    */
    return;
}

void PrefixSumBlending_GPU::reduceColor(uint32_t n_pixels, uint32_t n_samples_per_pixel, const float *d_alpha_in, const float *d_weights_in, const float3 *d_colors_in, float3 *d_img_out)
{
    /*
        Task 2:
        TODO: Perform a parallel reduction of all weighted colors to compute a single color value per pixel (see Equation (1) in assignment sheet)
    */
    return;
}