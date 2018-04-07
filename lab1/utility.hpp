#include <cstdint>
#include <fstream>
#include <vector>

namespace BMP {
    void read_bmp(std::string file_name, std::vector<uint8_t>& img) 
    {
        const std::ifstream::off_type bmp_file_header_size = 14;
        const std::ifstream::off_type dib_header_size = 40;

        std::ifstream f(file_name, std::ifstream::in | std::ifstream::binary);

        f.seekg(bmp_file_header_size + 4, f.beg);

        int width, height;
        f.read(reinterpret_cast<char*>(&width), 4);
        f.read(reinterpret_cast<char*>(&height), 4);

        f.seekg(bmp_file_header_size + dib_header_size, f.beg);

        img.resize(width * height * 3);
        uint8_t r, g, b;
        int idx;

        for (int j = height - 1; j >= 0; j--) {
                for (int i = 0; i < width; i++) {
                f.read(reinterpret_cast<char*>(&b), 1);
                f.read(reinterpret_cast<char*>(&g), 1);
                f.read(reinterpret_cast<char*>(&r), 1);

                idx = i + j * width;
                img[3 * idx] = static_cast<uint8_t>(r);
                img[3 * idx + 1] = static_cast<uint8_t>(g);
                img[3 * idx + 2] = static_cast<uint8_t>(b);
            }
        }

        f.close();
    }
}