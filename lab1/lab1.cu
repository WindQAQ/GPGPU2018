#include <cmath>
#include <cstdint>
#include <algorithm>

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#include "lab1.h"
#include "utility.hpp"

#define dt 0.05f
#define RGB2Y(r, g, b) ( +0.299 * (r) + 0.587 * (g) + 0.114 * (b) )
#define RGB2U(r, g, b) ( -0.169 * (r) - 0.331 * (g) + 0.500 * (b) + 128.0 )
#define RGB2V(r, g, b) ( +0.500 * (r) - 0.419 * (g) - 0.081 * (b) + 128.0 )

static const std::string BACKGROUND_FILE_NAME = "./aladdin_lamp.bmp";
static const unsigned W = 512;
static const unsigned H = 512;
static const unsigned NFRAME = 240;
static const int SIZE = W * H;
static const int THREADS_PER_BLOCK = 256;
static const int NUMBER_OF_BLOCKS = SIZE / THREADS_PER_BLOCK;
static const int NUMBER_OF_ITERATIONS = 50;
static const float DIFFUSION = 0.00001f;

enum class Boundary {
	U, 
	V, 
	D
};

__device__ inline bool not_boundary(int x, int y)
{
	// the boundary value is the `wall`
	return (x > 0) && (y > 0) && (x < W - 1) && (y < H - 1);
}

__device__ inline float clamp(float x, float _min, float _max)
{
	return fminf(fmaxf(x, _min), _max);
}

__device__ inline float bilinear_interpolate(float x, float y, const float *s)
{
	int px0, px1, py0, py1;
	float dx0, dx1, dy0, dy1;

	px0 = static_cast<int>(x), py0 = static_cast<int>(y);
	px1 = px0 + 1, py1 = py0 + 1;
	dx1 = x - px0, dy1 = y - py0;
	dx0 = 1.0 - dx1; dy0 = 1.0 - dy1;

	return dx0 * (dy0 * s[px0 + W * py0] + dy1 * s[px0 + W * py1]) +
			dx1 * (dy0 * s[px1 + W * py0] + dy1 * s[px1 + W * py1]);
}

__global__ void density_to_yuv(const float *d, uint8_t *yuv)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	const float r = clamp(255.0f - d[idx], 0.0f, 255.0f),
				g = 250.0f,
				b = 255.0f;

	if (fabsf(d[idx]) <= 1e-6) return;

	yuv[idx] = static_cast<uint8_t>(RGB2Y(r, g, b));

	if (x % 2 == 1 && y % 2 == 1) {
		const int sx = x / 2, sy = y / 2;
		const int sidx = sx + sy * W / 2;
		yuv[SIZE + sidx] = static_cast<uint8_t>(RGB2U(r, g, b));
		yuv[SIZE + SIZE / 4 + sidx] = static_cast<uint8_t>(RGB2V(r, g, b));
	}
}

template <typename T>
__global__ void rgb_to_yuv(const T *rgb, uint8_t *yuv)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	const T r = rgb[3 * idx + 0],
			g = rgb[3 * idx + 1],
			b = rgb[3 * idx + 2];

	yuv[idx] = static_cast<uint8_t>(RGB2Y(r, g, b));

	if (x % 2 == 1 && y % 2 == 1) {
		const int sx = x / 2, sy = y / 2;
		const int sidx = sx + sy * W / 2;
		yuv[SIZE + sidx] = static_cast<uint8_t>(128);
		yuv[SIZE + SIZE / 4 + sidx] = static_cast<uint8_t>(128);
	}
}

__global__ void add_force(float *d, const float *s)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	if (not_boundary(x, y)) {
		d[idx] += dt * s[idx];
	}
}

__global__ void transport(float *d, const float *d0, const float *u, const float *v, const float dissipation)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	float nx, ny;

	if (not_boundary(x, y)) {
		// trace practicle
		nx = x - dt * u[idx];
		ny = y - dt * v[idx];

		// clip value out of boundary
		nx = clamp(nx, 0.5, W - 1.5);
		ny = clamp(ny, 0.5, H - 1.5);

		// bi-linear interpolate
		d[idx] = bilinear_interpolate(nx, ny, d0);
		d[idx] *= dissipation;
	}
}

__global__ void get_divergence(const float *u, const float *v, float *div)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	if (not_boundary(x, y)) {
		// calculate gradient from neighbors (difference)
		div[idx] = -0.5 * ( (u[idx + 1] - u[idx - 1]) +
					(v[x + (y + 1) * W] - v[x + (y - 1) * W]) );
	}
}

__global__ void linear_solve_project(const float *div, const float *p0, float *p)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	if (not_boundary(x, y)) {
		p[idx] = (p0[idx - 1] + p0[idx + 1] + p0[x + (y - 1) * W] + 
					p0[x + (y + 1) * W] + div[idx]) * 0.25;
	}
}

__global__ void linear_solve_diffuse(const float *d0, float *d, const float diffusion)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	if (not_boundary(x, y)) {
		d[idx] = (d0[idx] + diffusion * (d0[idx - 1] + d0[idx + 1] + d0[x + (y - 1) * W] + d0[x + (y - 1) * W])) / (4.0f * diffusion + 1.0f);
	}
}

__global__ void update_velocity(float *u, float *v, const float *p)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	if (not_boundary(x, y)) {
		u[idx] -= 0.5 * (p[idx + 1] - p[idx - 1]);
		v[idx] -= 0.5 * (p[x + (y + 1) * W] - p[x + (y - 1) * W]);
	}
}

__global__ void set_boundary(float *p, Boundary mode)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	if (x == 0) {
		switch (mode) {
			case Boundary::U: p[idx] = -p[idx + 1]; break;
			case Boundary::V: case Boundary::D: p[idx] = p[idx + 1]; break;
		}
	}
	else if (y == 0) {
		switch (mode) {
			case Boundary::U: case Boundary::D: p[idx] = p[x + (y + 1) * W]; break;
			case Boundary::V: p[idx] = -p[x + (y + 1) * W]; break;
		}
	}
	else if (x == W - 1) {
		switch (mode) {
			case Boundary::U: p[idx] = -p[idx - 1]; break;
			case Boundary::V: case Boundary::D: p[idx] = p[idx - 1]; break;
		}		
	}
	else if (y == H - 1) {
		switch (mode) {
			case Boundary::U: case Boundary::D: p[idx] = p[x + (y - 1) * W]; break;
			case Boundary::V: p[idx] = -p[x + (y - 1) * W]; break;
		}
	}
}

__global__ void init_source(float *u, float *v, float *d)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	int dx = x - W / 2, dy = -(y - H / 2);
	int distance = dx * dx + dy * dy;

	if (distance <= 2500) {
		u[idx] = 10.0 * (dy - dx);
		v[idx] = 10.0 * (-dx - dy);
	}

	// u[idx] = 50.0;

	dx = x - W / 2 + 100, dy = -(y - H / 2);
	distance = dx * dx + dy * dy;

	if (distance <= 900)  d[idx] = 100.0;

	dx = x - W / 2 - 100, dy = -(y - H / 2);
	distance = dx * dx + dy * dy;

	if (distance <= 900)  d[idx] = 100.0;
}

__global__ void compute_curl(const float *u, const float *v, float *curl)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	if (not_boundary(x, y)) {
		curl[idx] = fabsf(0.5 * (v[idx + 1] - v[idx - 1] - u[x + (y + 1) * W] + u[x + (y - 1) * W]));
	}
}

__global__ void vc_update(float *u0, float *v0, const float *curl, const float *u, const float *v)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	if (not_boundary(x, y)) {
		float dw_dx = (curl[idx + 1] - curl[idx - 1]) * 0.5;
		float dw_dy = (curl[x + (y + 1) * W] - curl[x + (y - 1) * W]) * 0.5;

		float len = sqrtf(dw_dx * dw_dx + dw_dy * dw_dy) + 1e-6;

		dw_dx /= len, dw_dy /= len;

		float c = (u[x + (y + 1) * W] - u[x + (y - 1) * W]) * 0.5 -
					(v[idx + 1] - v[idx - 1]) * 0.5;

		u0[idx] = dw_dy * -c;
		v0[idx] = dw_dx * c;
	}
}

__host__ void compute_buoyancy(const float *d, float *v)
{
	float s = thrust::reduce(thrust::device, d, d + SIZE, 0.0f);

	s /= SIZE;

	auto f = [=] __device__ (float x) {
		return -100.0 * (0.000625 * x + 0.025 * (x - s));
	};

	thrust::transform(thrust::device, d, d + SIZE, v, f);

	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (v, Boundary::V);
}

__host__ void project(float *u, float *v, float *p, float *p0, float *div)
{
	get_divergence <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (u, v, div);
	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (div, Boundary::D);

	cudaMemset(p, 0, SIZE * sizeof(float));
	cudaMemset(p0, 0, SIZE * sizeof(float));

	for (int i = 0; i < NUMBER_OF_ITERATIONS; i++) {
		linear_solve_project <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (div, p0, p);
		
		set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (p, Boundary::D);

		cudaMemcpy(p0, p, SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
	}

	update_velocity <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (u, v, p);
	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (u, Boundary::U); 
	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (v, Boundary::V);
}

__host__ void diffuse(float *d0, float *d, const float diffusion, Boundary mode)
{
	cudaMemset(d, 0, SIZE * sizeof(float));

	for (int i = 0; i < NUMBER_OF_ITERATIONS; i++) {
		linear_solve_diffuse <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (d0, d, diffusion);

		set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (d, mode);

		cudaMemcpy(d0, d, SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
	}
}

__host__ void advect(float *d, const float *d0, const float *u, const float *v, const float dissipation, Boundary mode)
{
	transport <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (d, d0, u, v, dissipation);
	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (d, mode);
}

__host__ void vorticity_confine(float *u, float *v, float *curl, const float *u0, const float *v0)
{
	compute_curl <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (u0, v0, curl);
	vc_update <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (u, v, curl, u0, v0);
	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (u, Boundary::U);
	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (v, Boundary::V);
}

struct Lab1VideoGenerator::Impl {
	int t = 0;
	float *u, *v, *p;
	float *u0, *v0, *p0;
	float *u_source, *v_source;
	float *u_temp, *v_temp;

	float *div;
	float *curl;

	float *d, *d0;
	float *d_source;

	uint8_t *background_img;
};

Lab1VideoGenerator::Lab1VideoGenerator(): impl(new Impl) {
	cudaMalloc(&(impl->u), SIZE * sizeof(float));
	cudaMalloc(&(impl->v), SIZE * sizeof(float));
	cudaMalloc(&(impl->p), SIZE * sizeof(float));
	cudaMalloc(&(impl->d), SIZE * sizeof(float));

	cudaMalloc(&(impl->u0), SIZE * sizeof(float));
	cudaMalloc(&(impl->v0), SIZE * sizeof(float));
	cudaMalloc(&(impl->p0), SIZE * sizeof(float));
	cudaMalloc(&(impl->d0), SIZE * sizeof(float));

	cudaMalloc(&(impl->u_source), SIZE * sizeof(float));
	cudaMalloc(&(impl->v_source), SIZE * sizeof(float));
	cudaMalloc(&(impl->d_source), SIZE * sizeof(float));

	cudaMalloc(&(impl->div), SIZE * sizeof(float));
	cudaMalloc(&(impl->curl), SIZE * sizeof(float));

	cudaMalloc(&(impl->u_temp), SIZE * sizeof(float));
	cudaMalloc(&(impl->v_temp), SIZE * sizeof(float));

	cudaMalloc(&(impl->background_img), SIZE * 3 * sizeof(uint8_t));

	std::vector <uint8_t> img_vec;
	BMP::read_bmp(BACKGROUND_FILE_NAME, img_vec);
	cudaMemcpy(impl->background_img, img_vec.data(), SIZE * 3 * sizeof(uint8_t), cudaMemcpyHostToDevice);

	cudaMemset(impl->u0, 0, SIZE * sizeof(float));
	cudaMemset(impl->v0, 0, SIZE * sizeof(float));
	cudaMemset(impl->d0, 0, SIZE * sizeof(float));

	cudaMemset(impl->u_source, 0, SIZE * sizeof(float));
	cudaMemset(impl->v_source, 0, SIZE * sizeof(float));
	cudaMemset(impl->d_source, 0, SIZE * sizeof(float));

	init_source <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->u_source, impl->v_source, impl->d_source);
}

Lab1VideoGenerator::~Lab1VideoGenerator() {
	cudaFree(impl->u), cudaFree(impl->u0), cudaFree(impl->u_source), cudaFree(impl->u_temp);
	cudaFree(impl->v), cudaFree(impl->v0), cudaFree(impl->v_source), cudaFree(impl->v_temp);
	cudaFree(impl->p), cudaFree(impl->p0);
	cudaFree(impl->d), cudaFree(impl->d0), cudaFree(impl->d_source);

	cudaFree(impl->div);
	cudaFree(impl->curl);
}

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
	/* velocity step */
	add_force <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->u0, impl->u_source);
	add_force <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->v0, impl->v_source);

	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->u0, Boundary::U);
	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->v0, Boundary::V);

	// add buoyancy
	compute_buoyancy(impl->d0, impl->v_temp);
	add_force <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->v0, impl->v_temp);

	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->v0, Boundary::V);

	// vorticity confinement
	vorticity_confine(impl->u_temp, impl->v_temp, impl->curl, impl->u0, impl->v0);
	add_force <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->u0, impl->u_temp);
	add_force <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->v0, impl->v_temp);

	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->u0, Boundary::U);
	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->v0, Boundary::V);

	// diffuse
	diffuse(impl->u0, impl->u, DIFFUSION, Boundary::U);
	diffuse(impl->v0, impl->v, DIFFUSION, Boundary::V);

	project(impl->u0, impl->v0, impl->p, impl->p0, impl->div);

	// transport
	advect(impl->u, impl->u0, impl->u0, impl->v0, 1.0, Boundary::U);
	advect(impl->v, impl->v0, impl->u0, impl->v0, 1.0, Boundary::V);

	project(impl->u, impl->v, impl->p, impl->p0, impl->div);

	/* end velocity step */

	/* scalar step */
	add_force <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->d0, impl->d_source);

	// diffuse
	diffuse(impl->d0, impl->d, DIFFUSION, Boundary::D);

	// transport
	advect(impl->d, impl->d0, impl->u, impl->v, 0.995, Boundary::D);

	/* end scalar step */

	cudaDeviceSynchronize();

	cudaMemcpy(impl->u0, impl->u, SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(impl->v0, impl->v, SIZE * sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(impl->d0, impl->d, SIZE * sizeof(float), cudaMemcpyDeviceToDevice);

	// copy to frame
	cudaMemset(yuv, 255, SIZE);
	cudaMemset(yuv + SIZE, 128, SIZE / 2);

	rgb_to_yuv <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->background_img, yuv);

	density_to_yuv <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->d, yuv);

	++(impl->t);
}
