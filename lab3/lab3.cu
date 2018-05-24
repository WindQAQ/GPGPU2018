#include "lab3.h"
#include <cstdio>

__constant__ int dir[][2] = {
	{0, -1},
	{-1, 0},
	{+1, 0},
	{0, +1}
};

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__device__ __host__ inline bool in_boundary(const int w, const int h, const int x, const int y)
{
	return x >= 0 && x < w && y >= 0 && y < h;
};

__device__ __host__ inline int to_1dindex(const int w, const int x, const int y)
{
	return w * y + x;
}

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt * yt + xt;
	if (yt < ht && xt < wt && mask[curt] > 127.0f) {
		const int yb = oy + yt, xb = ox + xt;
		const int curb = wb * yb + xb;
		if (0 <= yb && yb < hb && 0 <= xb && xb < wb) {
			output[3 * curb + 0] = target[3 * curt + 0];
			output[3 * curb + 1] = target[3 * curt + 1];
			output[3 * curb + 2] = target[3 * curt + 2];
		}
	}
}

namespace baseline {

	__global__ void CalculateFixed(
		const float *background,
		const float *target,
		const float *mask,
		float *fixed,
		const int wb, const int hb, const int wt, const int ht,
		const int oy, const int ox
	)
	{
		const int yt = blockIdx.y * blockDim.y + threadIdx.y;
		const int xt = blockIdx.x * blockDim.x + threadIdx.x;
		const int center_idx = to_1dindex(wt, xt, yt);

		if (in_boundary(wt, ht, xt, yt)) {
			int neighbor = 0, neighbor_idx, background_idx;
			int xb, yb, xn, yn;
			float r = 0.0f, g = 0.0f, b = 0.0f;

			for (int i = 0; i < 4; i++) {
				xn = xt + dir[i][0], yn = yt + dir[i][1];
				xb = xn + ox, yb = yn + oy;
				neighbor_idx = to_1dindex(wt, xn, yn);
				background_idx = to_1dindex(wb, xb, yb);
				if (in_boundary(wt, ht, xn, yn)) {
					r -= target[3 * neighbor_idx + 0];
					g -= target[3 * neighbor_idx + 1];
					b -= target[3 * neighbor_idx + 2];

					if (mask[neighbor_idx] <= 127.0f && in_boundary(wb, hb, xb, yb)) {
						r += background[3 * background_idx + 0],
						g += background[3 * background_idx + 1],
						b += background[3 * background_idx + 2];
					}
					neighbor++;
				}
				else if (in_boundary(wb, hb, xb, yb)) {
					r += background[3 * background_idx + 0],
					g += background[3 * background_idx + 1],
					b += background[3 * background_idx + 2];
				}
			}

			r += neighbor * target[3 * center_idx + 0];
			g += neighbor * target[3 * center_idx + 1];
			b += neighbor * target[3 * center_idx + 2];

			fixed[3 * center_idx + 0] = r;
			fixed[3 * center_idx + 1] = g;
			fixed[3 * center_idx + 2] = b;
		}
	}

	__global__ void PoissonImageCloningIteration(
		const float *fixed,
		const float *mask,
		const float *input,
		float *output,
		const int wt, const int ht
	)
	{
		const int yt = blockIdx.y * blockDim.y + threadIdx.y;
		const int xt = blockIdx.x * blockDim.x + threadIdx.x;
		const int center_idx = to_1dindex(wt, xt, yt);

		if (in_boundary(wt, ht, xt, yt) && mask[center_idx] > 127.0f) {
			int neighbor_idx;
			int xn, yn;
			float r = fixed[3 * center_idx + 0], g = fixed[3 * center_idx + 1], b = fixed[3 * center_idx + 2];

			for (int i = 0; i < 4; i++) {
				xn = xt + dir[i][0], yn = yt + dir[i][1];
				neighbor_idx = to_1dindex(wt, xn, yn);
				if (in_boundary(wt, ht, xn, yn)) {
					if (mask[neighbor_idx] > 127.0f) {
						r += input[3 * neighbor_idx + 0];
						g += input[3 * neighbor_idx + 1];
						b += input[3 * neighbor_idx + 2];
					}
				}
			}

			output[3 * center_idx + 0] = r / 4;
			output[3 * center_idx + 1] = g / 4;
			output[3 * center_idx + 2] = b / 4;
		}
	}

	void PoissonImageCloning(
		const float *background,
		const float *target,
		const float *mask,
		float *output,
		const int wb, const int hb, const int wt, const int ht,
		const int oy, const int ox
	)
	{
		float *fixed, *buf1, *buf2;
		cudaMalloc(&fixed, 3 * wt * ht * sizeof(float));
		cudaMalloc(&buf1, 3 * wt * ht * sizeof(float));
		cudaMalloc(&buf2, 3 * wt * ht * sizeof(float));

		// initialize the iteration
		dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);
		CalculateFixed <<<gdim, bdim>>> (
			background, target, mask, fixed,
			wb, hb, wt, ht, oy, ox
		);
		cudaMemcpy(buf1, target, 3 * wt * ht * sizeof(float), cudaMemcpyDeviceToDevice);

		// iterate
		for (int i = 0; i < 10000; ++i) {
			PoissonImageCloningIteration <<<gdim, bdim>>> (
				fixed, mask, buf1, buf2, wt, ht
			);
			PoissonImageCloningIteration <<<gdim, bdim>>> (
				fixed, mask, buf2, buf1, wt, ht
			);
		}

		// copy the image back
		cudaMemcpy(output, background, 3 * wb * hb * sizeof(float), cudaMemcpyDeviceToDevice);
		SimpleClone <<<gdim, bdim>>> (
			background, buf1, mask, output,
			wb, hb, wt, ht, oy, ox
		);

		// clean up
		cudaFree(fixed);
		cudaFree(buf1);
		cudaFree(buf2);
	}
}

void PoissonImageCloning (
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	// cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	// SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
	// 	background, target, mask, output,
	// 	wb, hb, wt, ht, oy, ox
	// );

	baseline::PoissonImageCloning(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
	);
}
