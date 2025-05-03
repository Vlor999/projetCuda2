
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
    uint32_t numberPixels = dimensions.x * dimensions.y;

    // example: allocate additional buffer (malloc not relevant for timing)
    CUDA_CHECK_THROW(cudaMalloc(&_d_weights, sizeof(float) * numberPixels * samples_per_pixel));

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

__global__ void prefixSumWeightsIteration(const float* alpha_in, float* weights_out, uint32_t numberPixels, uint32_t n_samples_per_pixel) {
    const uint32_t pixel = blockIdx.x;
    const uint32_t thread = threadIdx.x;
    const uint32_t numberthreads = blockDim.x;

    if (pixel >= numberPixels){
        return;
    }

    extern __shared__ float sharedMemo[];
    float* T = sharedMemo;

    for (uint32_t i = thread; i < n_samples_per_pixel; i += numberthreads){
        T[i] = 1.0f - alpha_in[pixel * n_samples_per_pixel + i];
    }

    __syncthreads();

    for (uint32_t s = 1; s < n_samples_per_pixel; s <<= 1) {
        uint32_t pos = (thread + 1) * s << 1 - 1;
        if (pos < n_samples_per_pixel) {
            T[pos] *= T[pos - s];
        }
        __syncthreads();
    }

    if (thread == 0) {
        T[n_samples_per_pixel - 1] = 1.0f;
    }
    __syncthreads();

    for (uint32_t s = n_samples_per_pixel >> 1; s >= 1; s >>= 1) {
        uint32_t i = (thread + 1) * s << 1 - 1;
        if (i < n_samples_per_pixel) {
            float temp = T[i - s];
            T[i - s] = T[i];
            T[i] *= temp;
        }
        __syncthreads();
    }

    for (uint32_t i = thread; i < n_samples_per_pixel; i += numberthreads) {
        weights_out[pixel * n_samples_per_pixel + i] = T[i];
    }
}

__global__ void reduceColorIteration(const float *alpha_in, const float *weights_in, const float3 *colors_in, float3 *img_out, uint32_t numberPixels, uint32_t n_samples_per_pixel){
    uint32_t pixel = blockIdx.x;
    if (pixel >= numberPixels){
        return;
    }
    uint32_t thread = threadIdx.x;
    uint32_t numberthreads = blockDim.x;


    extern __shared__ float3 sharedMem[];
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);

    for (uint32_t i = thread; i < n_samples_per_pixel; i += numberthreads) {
        uint32_t idx = pixel * n_samples_per_pixel + i;
        float ai = alpha_in[idx];
        float ti = weights_in[idx];
        float3 ci = colors_in[idx];
        sum += ci * ai * ti;
    }

    sharedMem[thread] = sum;
    __syncthreads();

    for (uint32_t s = numberthreads >> 1; s > 0; s >>= 1) {
        if (thread < s){
            sharedMem[thread] += sharedMem[thread + s];
        }
        __syncthreads();
    }

    if (thread == 0){
        img_out[pixel] = sharedMem[0];
    }
}

void PrefixSumBlending_GPU::prefixSumWeights(uint32_t numberPixels, uint32_t n_samples_per_pixel, const float *d_alpha_in, float *d_weights_out){
    // Equation : 2
    const uint32_t threads = BLOCK;
    const uint32_t blocks = numberPixels;
    size_t shared_mem_size = sizeof(float) * n_samples_per_pixel;

    prefixSumWeightsIteration<<<blocks, threads, shared_mem_size>>>(d_alpha_in, d_weights_out, numberPixels, n_samples_per_pixel);
    CUDA_SYNC_CHECK_THROW();
}

void PrefixSumBlending_GPU::reduceColor(uint32_t numberPixels, uint32_t n_samples_per_pixel, const float *d_alpha_in, const float *d_weights_in, const float3 *d_colors_in, float3 *img_out){
    // Equation : 1
    const uint32_t threads = BLOCK;
    const uint32_t blocks = numberPixels;
    size_t shared_mem_size = sizeof(float3) * threads;

    reduceColorIteration<<<blocks, threads, shared_mem_size>>>(d_alpha_in, d_weights_in, d_colors_in, img_out, numberPixels, n_samples_per_pixel);
    CUDA_SYNC_CHECK_THROW();
}