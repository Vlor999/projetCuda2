
#include "prefixsum.h"
#include "helper/helper_math.h"

#include <numeric>
#include <functional>

void PrefixSumBlending_CPU::run(Dataset &data, std::vector<float3> &img_out)
{
    std::vector<float> transmittance(data._alphas_unpacked.size());
    this->prefixSumWeights(data._n_pixels, data._samples_per_pixel, data._alphas_unpacked, transmittance);
    
    this->reduceColor(data._n_pixels, data._samples_per_pixel, data._alphas_unpacked, transmittance, data._colors_unpacked, img_out);
}

void PrefixSumBlending_CPU::prefixSumWeights(uint32_t n_pixels, uint32_t n_samples_per_pixel, const std::vector<float>& alpha_in, std::vector<float>& weights_out)
{
    /*
        Compute the parallel prefix sum for the transmittance values T_i
        See Equation (2) in assignment sheet
    */
    for (uint32_t pix_idx = 0; pix_idx < n_pixels; pix_idx++)
    {
        float T = 1.0f;
        for (uint32_t i = 0; i < n_samples_per_pixel; i++)
        {
            uint32_t sample_idx = pix_idx * n_samples_per_pixel + i;
            float alpha = alpha_in[sample_idx];
            weights_out[sample_idx] = T;
            T *= (1.0f - alpha);
        }
    }
}

void PrefixSumBlending_CPU::reduceColor(uint32_t n_pixels, uint32_t n_samples_per_pixel, const std::vector<float>& alpha_in, const std::vector<float>& weights_in, const std::vector<float3>& colors_in, std::vector<float3>& img_out)
{
    /*
        Perform a parallel reduction of all weighted colors to compute a single color value per pixel 
        See Equation (1) in assignment sheet
    */
    for (uint32_t pix_idx = 0; pix_idx < n_pixels; pix_idx++)
    {
        float3 out_color = make_float3(0.0f);
        for (uint32_t i = 0; i < n_samples_per_pixel; i++)
        {
            uint32_t sample_idx = pix_idx * n_samples_per_pixel + i;
            float3 color = colors_in[sample_idx];
            float alpha = alpha_in[sample_idx];
            float T = weights_in[sample_idx];

            float3 weighted_color = color * T * alpha;
            out_color += weighted_color;
        }

        img_out[pix_idx] = out_color;
    }
}