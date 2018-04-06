#include <cmath>
#include <cstdint>
#include <algorithm>
#include "lab1.h"

static const unsigned W = 512;
static const unsigned H = 512;
static const unsigned NFRAME = 240;
static const int SIZE = W *H;
static const int THREADS_PER_BLOCK = 256;
static const int NUMBER_OF_BLOCKS = SIZE / THREADS_PER_BLOCK;
static const float dt = 1.0f;

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

	px0 = __float2int_rd(x), py0 = __float2int_rd(y);
	px1 = px0 + 1, py1 = py0 + 1;
	dx1 = x - px0, dy1 = y - py0;
	dx0 = 1.0 - dx1; dy0 = 1.0 - dy1;

	return dx0 * (dy1 * s[px0 + W * py0] + dy1 * s[px0 + W * py1]) +
			dx1 * (dy0 * s[px1 + W * py0] + dy1 * s[px1 + W * py1]);
}

__global__ void add_force(float *d, const float *s)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	if (not_boundary(x, y)) {
		// d[idx] += dt * s[idx];
		d[idx] += 1.0 * s[idx];
	}
}

__global__ void transport(float *d, float *d0, float *u, float *v, const float dissipation)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	float nx, ny;

	if (not_boundary(x, y)) {
		// trace practicle
		nx = x - 1.0 * u[idx];
		ny = y - 1.0 * v[idx];

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
		div[idx] = -0.5 * ( (u[idx + 1] - u[idx - 1]) / (W - 2) + 
					(v[x + (y - 1) * W] - v[x + (y + 1) * W]) / (H - 2) );
	}
}

__global__ void linear_solve(const float *u, const float *v, const float *div, const float *p0, float *p)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	if (not_boundary(x, y)) {
		p[idx] = (p0[idx - 1] + p0[idx + 1] + p0[x + (y - 1) * W] + 
					p0[x + (y + 1) * W] + div[idx]) * 0.25;
	}
}

__global__ void update_velocity(float *u, float *v, const float *p)
{
	const int idx = threadIdx.x + blockIdx.x * blockDim.x;
	const int x = idx % W, y = idx / W;

	if (not_boundary(x, y)) {
		u[idx] -= 0.5 * (p[idx + 1] - p[idx - 1]) * (W - 2);
		v[idx] -= 0.5 * (p[x + (y + 1) * W] - p[x + (y - 1) * W]) * (H - 2);
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

	int dx = x - W / 2, dy = y - H / 2;

	int distance = dx * dx + dy * dy;
	if (distance <= 10000) {
		d[idx] = 10.0;
	}

	u[idx] = 2 * dy / H;
	v[idx] = 2 * dx / W;
}

__host__ void project(float *u, float *v, float *p, float *p0, float *ptemp, float *div)
{
	get_divergence <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (u, v, div);

	cudaMemset(p, 0, SIZE * sizeof(float));

	for (int i = 0; i < 80; i++) {
		linear_solve <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (u, v, div, p0, p);

		ptemp = p0; p0 = p; p = ptemp;
		
		set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (p, Boundary::D);
	}

	update_velocity <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (u, v, p);
	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (u, Boundary::U); 
	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (v, Boundary::V);
}

struct Lab1VideoGenerator::Impl {
	int t = 0;
	float *u, *v, *p;
	float *u0, *v0, *p0;
	float *utemp, *vtemp, *ptemp;
	float *u_source, *v_source;

	float *div;

	float *d, *d0, *dtemp;
	float *d_source;
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

	cudaMemset(impl->u0, 0, SIZE * sizeof(float));
	cudaMemset(impl->v0, 0, SIZE * sizeof(float));
	cudaMemset(impl->d0, 0, SIZE * sizeof(float));

	cudaMemset(impl->u_source, 0, SIZE * sizeof(float));
	cudaMemset(impl->v_source, 0, SIZE * sizeof(float));
	cudaMemset(impl->d_source, 0, SIZE * sizeof(float));
}

Lab1VideoGenerator::~Lab1VideoGenerator() {
	cudaFree(impl->u), cudaFree(impl->u0), cudaFree(impl->u_source);
	cudaFree(impl->v), cudaFree(impl->v0), cudaFree(impl->v_source);
	cudaFree(impl->p), cudaFree(impl->p0);
	cudaFree(impl->d), cudaFree(impl->d0), cudaFree(impl->d_source);

	cudaFree(impl->div);
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
	float *m = new float[SIZE];
	uint8_t *n = new uint8_t[SIZE];

	if (impl->t == 0) {
		init_source <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->u_source, impl->v_source, impl->d_source);
	}

	// velocity step
	add_force <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->u0, impl->u_source);
	add_force <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->v0, impl->u_source);

	add_force <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->d0, impl->d_source);

	transport <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->u, impl->u0, impl->u0, impl->v0, 1.0);
	transport <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->v, impl->v0, impl->u0, impl->v0, 1.0);

	transport <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->d, impl->d0, impl->u, impl->v, 0.995);

	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->u, Boundary::U);
	set_boundary <<<NUMBER_OF_BLOCKS, THREADS_PER_BLOCK>>> (impl->v, Boundary::V);

	project(impl->u, impl->v, impl->p, impl->p0, impl->ptemp, impl->div);

	cudaMemcpy(m, impl->d, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	std::transform(m, m + SIZE, n, [](uint8_t a) -> float {
		return (uint8_t) std::max(std::min((float)a, 255.0f), 0.0f);
	});

	cudaMemcpy(yuv, n, SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemset(yuv + SIZE, 128, SIZE / 2 * sizeof(uint8_t));

	delete [] m;
	delete [] n;

	impl->utemp = impl->u0; impl->u0 = impl->u; impl->u = impl->utemp;
	impl->vtemp = impl->v0; impl->v0 = impl->v; impl->v = impl->vtemp;
	impl->ptemp = impl->p0; impl->p0 = impl->p; impl->p = impl->ptemp;

	++(impl->t);
}
