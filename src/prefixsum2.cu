#include "prefixsum.h"
#include "helper/helper_math.h"
#include "helper/cuda_helper_host.h"

#define BLOCK 256 // 512 does not inprove anything 

PrefixSumBlending_GPU::PrefixSumBlending_GPU() 
{
}

PrefixSumBlending_GPU::~PrefixSumBlending_GPU() 
{
}

void PrefixSumBlending_GPU::setup(uint2 dimensions, uint32_t samples_per_pixel) {
    CUDA_SYNC_CHECK_THROW();
}

void PrefixSumBlending_GPU::finalize() 
{

}

__global__ void prefixAndColor(uint32_t n_pixels, uint32_t n_samples_per_pixel, const float *alpha, const float3 *colors, float3 *img_out) {
    uint32_t pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= n_pixels) return;

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    uint32_t base = pixel * n_samples_per_pixel;

    float acc = 1.0f;
    for (uint32_t i = 0; i < n_samples_per_pixel; ++i) {
        float a = alpha[base + i];
        float3 c = colors[base + i];
        sum += c * a * acc;
        acc *= (1.0f - a);
    }

    img_out[pixel] = sum;
}

void PrefixSumBlending_GPU::run(DatasetGPU &data, float3 *d_img_out)
{
    uint32_t grid = (data._n_pixels + BLOCK - 1) / BLOCK;
    prefixAndColor<<<grid, BLOCK>>>(data._n_pixels, data._samples_per_pixel, data._alphas, data._colors,d_img_out);

    CUDA_SYNC_CHECK_THROW();
}

