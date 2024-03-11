
#include "dataset.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <numeric>

#include <json/json.hpp>
using json = nlohmann::json;

#include "helper/cuda_helper_host.h"

template <typename T>
bool readFile(std::vector<T> &vec, std::filesystem::path data_dir, const char filename[])
{
    std::filesystem::path input_file = data_dir / std::filesystem::path(filename);
    std::ifstream stream(input_file.c_str(), std::ios::in | std::ios::binary);

    if (!stream)
        return false;

    stream.seekg(0, std::ios::end);
    size_t filesize = stream.tellg();
    stream.seekg(0, std::ios::beg);

    vec.resize(filesize / sizeof(T));
    stream.read((char *)vec.data(), filesize);

    return true;
}

bool Dataset::load(fs::path input_dir)
{
    std::ifstream frameinfo_file(input_dir / "frameinfo.json");
    if (!frameinfo_file.is_open())
    {
        std::cout << "Could not load frameinfo file!" << std::endl;
        return false;
    }
    json frameinfo_json_data = json::parse(frameinfo_file);

    _dimensions.x = frameinfo_json_data["width"].get<uint32_t>();
    _dimensions.y = frameinfo_json_data["height"].get<uint32_t>();
    _n_pixels = _dimensions.x * _dimensions.y;

    std::cout << "Loading data... " << std::endl;
    readFile(_counts, input_dir, "packed_counts.dat");
    assert(_counts.size() == _n_pixels && "Expecting one count per pixel");
    _total_packed_count = std::reduce(_counts.begin(), _counts.end());

    readFile(_colors_packed, input_dir, "colors_packed.dat");
    readFile(_alphas_packed, input_dir, "alphas_packed.dat");
    assert(_colors_packed.size() == _total_packed_count && _alphas_packed.size() == _total_packed_count && "Packed data not matching count");

    _offsets.resize(_n_pixels);
    std::exclusive_scan(_counts.begin(), _counts.end(), _offsets.begin(), 0);

    return true;
}

void Dataset::unpack()
{
    auto minmax_count = std::minmax_element(_counts.begin(), _counts.end());
    _samples_per_pixel = std::pow(2, std::ceil(std::log2(*minmax_count.second))); // next power of 2

    std::cout << "Unpacking input data: " << _total_packed_count << " -> " << _n_pixels << " x " << _samples_per_pixel << " ... " << std::endl;
    _colors_unpacked.resize(_n_pixels * _samples_per_pixel);
    _alphas_unpacked.resize(_n_pixels * _samples_per_pixel, 0.0f);

    for (uint32_t pixel_idx = 0; pixel_idx < _n_pixels; pixel_idx++)
    {
        uint32_t offset = _offsets[pixel_idx];
        uint32_t count = _counts[pixel_idx];

        for (uint32_t sample_idx = 0; sample_idx < count; sample_idx++)
        {
            uint32_t packed_idx = offset + sample_idx;
            uint32_t unpacked_idx = pixel_idx * _samples_per_pixel + sample_idx;

            _colors_unpacked[unpacked_idx] = _colors_packed[packed_idx];
            _alphas_unpacked[unpacked_idx] = _alphas_packed[packed_idx];
        }
    }
}

DatasetGPU Dataset::upload()
{
    DatasetGPU data_gpu;

    data_gpu._dimensions = _dimensions;
    data_gpu._n_pixels = _n_pixels;

    data_gpu._samples_per_pixel = _samples_per_pixel;
    data_gpu._colors = uploadVector(_colors_unpacked);
    data_gpu._alphas = uploadVector(_alphas_unpacked);

    return data_gpu;
}