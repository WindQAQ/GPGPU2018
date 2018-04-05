#include <cmath>
#include <cstdint>
#include "lab1.h"
#include "frame.h"

static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;

inline void writeFrame(uint8_t *yuv, const Frame& rgb)
{
	auto frame = toYUV(rgb, W, H);

	cudaMemcpy(yuv, frame.data(), frame.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
}

struct Lab1VideoGenerator::Impl {
	int t = 0;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
}

Lab1VideoGenerator::~Lab1VideoGenerator() {}

void Lab1VideoGenerator::get_info(Lab1VideoInfo &info) 
{
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};


void Lab1VideoGenerator::Generate(uint8_t *yuv) 
{
	Frame frame;
	frame.resize(H * W);

	writeFrame(yuv, frame);
	++(impl->t);
}
