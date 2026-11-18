#pragma once
#include <cuda_runtime.h>
#include <cstddef>

#define FORALL(i, n) \
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)

__device__ __forceinline__
uint64_t atomicOr(uint64_t* address, uint64_t val)
{
	using ull = unsigned long long;
	return atomicOr(reinterpret_cast<ull*>(address), static_cast<ull>(val));
}

__device__ __forceinline__
uint64_t atomicAnd(uint64_t* address, uint64_t val)
{
	using ull = unsigned long long;
	return atomicAnd(reinterpret_cast<ull*>(address), static_cast<ull>(val));
}

#define GPU(N) GPU_nblocks(N), GPU_nthreads(N)

constexpr unsigned int GPU_nthreads(int n) {
	if (n <= 32 * 32) return 32;
	if (n <= 64 * 64) return 64;
	if (n <= 128 * 128) return 128;
	if (n <= 256 * 256) return 256;
	if (n <= 512 * 512) return 512;
	return 1024;
}
constexpr unsigned int GPU_nblocks_for_nthreads(int n, unsigned int nt) {
	return n / nt + (n % nt ? 1 : 0);
}
constexpr unsigned int GPU_nblocks(int n) {
	return GPU_nblocks_for_nthreads(n, GPU_nthreads(n));
}
