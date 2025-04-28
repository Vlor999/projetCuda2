#include "prefixsum.h"
#include "helper/helper_math.h"
#include "helper/cuda_helper_host.h"

#define BLOCK 256
#define WARP_SIZE 32

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

__global__ void prefixAndColorMultiThread(uint32_t n_pixels, uint32_t n_samples_per_pixel, const float *alpha, const float3 *colors, float3 *img_out) {
    uint32_t warpId = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    uint32_t laneId = threadIdx.x % WARP_SIZE;

    if (warpId >= n_pixels) return;

    uint32_t base = warpId * n_samples_per_pixel;
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    float ti = 1.0f;
    uint32_t writtingPos = base;
    for (uint32_t i = laneId; i < n_samples_per_pixel; i += WARP_SIZE) {
        float ai = alpha[writtingPos];
        float3 ci = colors[writtingPos];
        
        sum += ci * ai * ti;
        ti *= (1.0f - ai);
        writtingPos++;
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        sum.x += __shfl_down_sync(0xffffffff, sum.x, offset);
        sum.y += __shfl_down_sync(0xffffffff, sum.y, offset);
        sum.z += __shfl_down_sync(0xffffffff, sum.z, offset);
    }

    if (laneId == 0) {
        img_out[warpId] = sum;
    }
}

void PrefixSumBlending_GPU::run(DatasetGPU &data, float3 *d_img_out)
{
    uint32_t warpsBlock = BLOCK / WARP_SIZE;
    uint32_t grid = (data._n_pixels + warpsBlock - 1) / warpsBlock;

    prefixAndColorMultiThread<<<grid, BLOCK>>>(data._n_pixels, data._samples_per_pixel, data._alphas, data._colors, d_img_out);

    CUDA_SYNC_CHECK_THROW();
}
