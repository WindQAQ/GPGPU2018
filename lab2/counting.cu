#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

const int NUM_THREADS = 128;

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct F: public thrust::unary_function<char, int> {
    __device__ __host__ int operator() (char a) { return (a == '\n')? 0: 1; }
};

template <typename T>
void print(T* a, const int size)
{
    thrust::device_ptr<T> p(a);
    thrust::device_vector<T> v(p, p + size);

    thrust::copy(v.begin(), v.end(), std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;
}

void CountPosition1(const char *text, int *pos, int text_size)
{
    F f;
    thrust::transform(thrust::device, text, text + text_size, pos, f);
    thrust::inclusive_scan_by_key(thrust::device, pos, pos + text_size, pos, pos);
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
            else if (idx == 0 || (text[idx - 1] == '\n' && text[idx] != '\n')) {
                for (int i = idx, k = 1; i < text_size && text[i] != '\n'; i++, k++) {
                    pos[i] = k;
                }
            }
        }
    }
}

void CountPosition2(const char *text, int *pos, int text_size)
{
    Naive::count <<<CeilDiv(text_size, NUM_THREADS), NUM_THREADS>>> (text, pos, text_size);
}
