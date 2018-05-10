#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

const int NUM_THREADS = 512;

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct F: public thrust::unary_function<char, int> {
    __device__ __host__ int operator() (char a) { return (a == '\n')? 0: 1; }
};

void CountPosition1(const char *text, int *pos, int text_size)
{
    F f;
    int *keys;
    cudaMalloc((void **)&keys, text_size * sizeof(int));
    thrust::transform(thrust::device, text, text + text_size, keys, f);
    cudaMemcpy(pos, keys, text_size * sizeof(int), cudaMemcpyDeviceToDevice);
    thrust::inclusive_scan_by_key(thrust::device, keys, keys + text_size, pos, pos);
    cudaFree(keys);
}

namespace Naive {
    // O(nk) method
    __global__ void count(const char *text, int *pos, const int text_size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < text_size) {
            if (text[idx] == '\n') {
                pos[idx] = 0;
            }
            else if (idx == 0 || text[idx - 1] == '\n') {
                for (int i = idx, k = 1; i < text_size && text[i] != '\n'; i++, k++) {
                    pos[i] = k;
                }
            }
        }
    }
}

namespace Speed {
    // O(nlogk) method: segmented scan
    // Reference: http://research.nvidia.com/sites/default/files/publications/nvr-2008-003.pdf
    __device__ int segscan_warp(int *ptr, int *hd, const unsigned int idx = threadIdx.x)
    {
        const unsigned int lane = idx & 31;

        if (lane >= 1) {
            ptr[idx] += (!hd[idx]) * ptr[idx - 1];
            hd[idx] |= hd[idx - 1];
        }
        if (lane >= 2) {
            ptr[idx] += (!hd[idx]) * ptr[idx - 2];
            hd[idx] |= hd[idx - 2];
        }
        if (lane >= 4) {
            ptr[idx] += (!hd[idx]) * ptr[idx - 4];
            hd[idx] |= hd[idx - 4];
        }
        if (lane >= 8) {
            ptr[idx] += (!hd[idx]) * ptr[idx - 8];
            hd[idx] |= hd[idx - 8];
        }
        if (lane >= 16) {
            ptr[idx] += (!hd[idx]) * ptr[idx - 16];
            hd[idx] |= hd[idx - 16];
        }

        return ptr[idx];
    }

    __device__ int segscan_block(int *ptr, int *hd, const unsigned int idx = threadIdx.x)
    {
        unsigned int warpid = idx >> 5;
        unsigned int warp_first = warpid << 5;
        unsigned int warp_last = warp_first + 31;

        bool warp_is_open = (hd[warp_first] == 0);
        __syncthreads();

        int val = segscan_warp(ptr, hd, idx);

        int warp_total = ptr[warp_last];

        int warp_flag = (hd[warp_last] != 0) || !warp_is_open;
        bool will_accumulate = warp_is_open && (hd[idx] == 0);

        __syncthreads();

        if (idx == warp_last) {
            ptr[warpid] = warp_total;
            hd[warpid] = warp_flag;
        }
        __syncthreads();

        if (warpid == 0) segscan_warp(ptr, hd, idx);

        __syncthreads();

        if (warpid != 0 && will_accumulate) val += ptr[warpid - 1];
        __syncthreads();

        ptr[idx] = val;
        __syncthreads();

        return val;
    }

    __global__ void segscan(const char *text, int *pos, const int text_size)
    {
        __shared__ int ptr[NUM_THREADS];
        __shared__ int hd[NUM_THREADS];
        const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int f;
        if (idx < text_size) {
            f = (text[idx] != '\n');
            hd[threadIdx.x] = (idx == 0 || (f && text[idx - 1] == '\n'));
            ptr[threadIdx.x] = f;
        }
        __syncthreads();

        segscan_block(ptr, hd);
        __syncthreads();

        if (idx < text_size) pos[idx] = f * ptr[threadIdx.x];
    }

    __device__ int scan_warp(int *ptr, const unsigned int idx = threadIdx.x)
    {
        const unsigned int lane = idx & 31;

        if (lane >= 1) ptr[idx] *= ptr[idx - 1];
        if (lane >= 2) ptr[idx] *= ptr[idx - 2];
        if (lane >= 4) ptr[idx] *= ptr[idx - 4];
        if (lane >= 8) ptr[idx] *= ptr[idx - 8];
        if (lane >= 16) ptr[idx] *= ptr[idx - 16];

        return ptr[idx];
    }

    __device__ int scan_block(int *ptr, const unsigned int idx = threadIdx.x)
    {
        const unsigned int lane = idx & 31;
        const unsigned int warpid = idx >> 5;

        int val = scan_warp(ptr, idx);
        __syncthreads();

        if (lane == 31) ptr[warpid] = ptr[idx];
        __syncthreads();

        if (warpid == 0) scan_warp(ptr, idx);
        __syncthreads();

        if (warpid > 0) val *= ptr[warpid -1];
        __syncthreads();

        ptr[idx] = val;
        __syncthreads();
        return val;
    }

    __global__ void regularize(const char *text, int *pos, const int text_size)
    {
        __shared__ int ptr[NUM_THREADS];
        __shared__ int prev;
        const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < text_size) ptr[threadIdx.x] = (text[idx] != '\n');
        if (threadIdx.x == 0) prev = (blockIdx.x == 0)? 0: pos[idx - 1];
        __syncthreads();

        scan_block(ptr);
        __syncthreads();

        if (idx < text_size) pos[idx] += ptr[threadIdx.x] * prev;
    }

    void count(const char *text, int *pos, const int text_size)
    {
        segscan <<<CeilDiv(text_size, NUM_THREADS), NUM_THREADS>>> (text, pos, text_size);
        regularize <<<CeilDiv(text_size, NUM_THREADS), NUM_THREADS>>> (text, pos, text_size);
    }
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    // Naive::count <<<CeilDiv(text_size, NUM_THREADS), NUM_THREADS>>> (text, pos, text_size);
    Speed::count(text, pos, text_size);
}