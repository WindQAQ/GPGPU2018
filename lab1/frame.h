#ifndef _frame_h
#define _frame_h

#include <cstdint>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>

static const int NUMBER_OF_CHANNELS = 3;
static const std::array<std::array<double, NUMBER_OF_CHANNELS>, NUMBER_OF_CHANNELS> M = {{
    {+0.299, +0.587, +0.114},
    {-0.169, -0.331, +0.500},
    {+0.500, -0.419, -0.081}
}};

using Pixel = std::array<uint8_t, NUMBER_OF_CHANNELS>;
using Frame = std::vector< std::array<uint8_t, NUMBER_OF_CHANNELS> >;

// https://stackoverflow.com/questions/9465815/rgb-to-yuv420-algorithm-efficiency
std::vector<uint8_t> toYUV(const Frame& rgb, const int W, const int H)
{
    // yuv layout: y -> u -> v
    std::vector<uint8_t> yuv;
    yuv.resize(W * H * 3 / 2);

    size_t y_pos = 0;
    size_t u_pos = W * H;
    size_t v_pos = u_pos + u_pos / 4;

    for (unsigned i = 0; i < H; i++) {
        if (!(i % 2)) {
            for (unsigned j = 0; j < W; j += 2) {
                auto& pix = rgb[i * W + j];

                yuv[y_pos++] = static_cast<uint8_t>(std::inner_product(pix.begin(), pix.end(), M[0].begin(), 0.0));
                yuv[u_pos++] = static_cast<uint8_t>(std::inner_product(pix.begin(), pix.end(), M[1].begin(), 128.0));
                yuv[v_pos++] = static_cast<uint8_t>(std::inner_product(pix.begin(), pix.end(), M[2].begin(), 128.0));

                auto& pix_next = rgb[(i + 1) * W + j];            

                yuv[y_pos++] = static_cast<uint8_t>(std::inner_product(pix_next.begin(), pix_next.end(), M[0].begin(), 0.0));
            }
        }
        else {
            for (unsigned j = 0; j < W; j++) {
                auto& pix = rgb[i * W + j];
    
                yuv[y_pos++] = static_cast<uint8_t>(std::inner_product(pix.begin(), pix.end(), M[0].begin(), 0.0));
            }
        }
    }

    return yuv;
}

#endif