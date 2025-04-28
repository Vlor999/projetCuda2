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

void PrefixSumBlending_GPU::setup(uint2 dimensions, uint32_t samples_per_pixel)
{
    uint32_t numberPixels = dimensions.x * dimensions.y;

    // example: allocate additional buffer (malloc not relevant for timing)
    CUDA_CHECK_THROW(cudaMalloc(&_d_weights, sizeof(float) * numberPixels * samples_per_pixel));
    CUDA_SYNC_CHECK_THROW();
}

void PrefixSumBlending_GPU::finalize()
{
    if (_d_weights){
        cudaFree(_d_weights);
    }
}

__global__ void prefixAndColorMultiThread(const float* alpha_in, const float3* colors_in, float3* img_out, uint32_t numberPixels, uint32_t n_samples_per_pixel) {
    const uint32_t pixel = blockIdx.x;
    const uint32_t thread = threadIdx.x;
    const uint32_t numberthreads = blockDim.x;

    if (pixel >= numberPixels) {
        return;
    }

    extern __shared__ float sharedMem[];
    float* T = sharedMem;
    float3* colorSums = (float3*)(sharedMem + n_samples_per_pixel);

    for (uint32_t i = thread; i < n_samples_per_pixel; i += numberthreads) {
        T[i] = 1.0f - alpha_in[pixel * n_samples_per_pixel + i];
    }

    __syncthreads();

    for (uint32_t s = 1; s < n_samples_per_pixel; s *= 2) {
        uint32_t i = (thread + 1) * s * 2 - 1;
        if (i < n_samples_per_pixel) {
            T[i] *= T[i - s];
        }
        __syncthreads();
    }

    if (thread == 0) {
        T[n_samples_per_pixel - 1] = 1.0f;
    }
    __syncthreads();

    for (uint32_t s = n_samples_per_pixel / 2; s >= 1; s /= 2) {
        uint32_t i = (thread + 1) * s * 2 - 1;
        if (i < n_samples_per_pixel) {
            float temp = T[i - s];
            T[i - s] = T[i];
            T[i] *= temp;
        }
        __syncthreads();
    }

    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    for (uint32_t i = thread; i < n_samples_per_pixel; i += numberthreads) {
        uint32_t idx = pixel * n_samples_per_pixel + i;
        float ai = alpha_in[idx];
        float ti = T[i];
        float3 ci = colors_in[idx];
        sum += ci * ai * ti;
    }

    if (thread < numberthreads) {
        colorSums[thread] = sum;
    }
    __syncthreads();

    for (uint32_t s = numberthreads / 2; s > 0; s >>= 1) {
        if (thread < s) {
            colorSums[thread] += colorSums[thread + s];
        }
        __syncthreads();
    }

    if (thread == 0) {
        img_out[pixel] = colorSums[0];
    }
}

void PrefixSumBlending_GPU::run(DatasetGPU &data, float3 *d_img_out)
{
    uint32_t threads = BLOCK;
    uint32_t blocks = data._n_pixels;
    size_t shared_mem_size = (data._samples_per_pixel * sizeof(float)) + (threads * sizeof(float3));

    prefixAndColorMultiThread<<<blocks, threads, shared_mem_size>>>(data._alphas, data._colors, d_img_out, data._n_pixels, data._samples_per_pixel);

    CUDA_SYNC_CHECK_THROW();
}