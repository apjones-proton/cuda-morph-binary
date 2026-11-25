#include "../include/morphology.h"
#include "cases.h"
#include "cpu_morphology.h"
#include <cuda_runtime.h>
#include <nppi.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <cassert>

int time_us_thislibrary(const Case& cs)
{
	const int w = cs.width;
	const int h = cs.height;
	const float r = cs.radius;
	const auto& input = cs.array;

	size_t max_alloc = w * h * sizeof(uint8_t);
	uint8_t* d_input; cudaMalloc(&d_input, max_alloc);
	uint8_t* d_output; cudaMalloc(&d_output, max_alloc);
	void* d_workspace; cudaMalloc(&d_workspace, max_alloc);
	cudaMemcpy(d_input, input.data(), w * h * sizeof(uint8_t), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();;
	auto before = std::chrono::high_resolution_clock::now();

	morph::dilate_binary(d_input, d_output, w, h, r, d_workspace);

	cudaDeviceSynchronize();;
	auto after = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);

	cudaFree(d_workspace);
	cudaFree(d_output);
	cudaFree(d_input);

	return duration.count();
}

int time_us_cpu_morphology(const Case& cs, int timeout_us)
{
	const int w = cs.width;
	const int h = cs.height;
	const float r = cs.radius;
	const auto& input = cs.array;

	size_t max_alloc = w * h * sizeof(uint8_t);
	std::vector<uint8_t> cpu_output(max_alloc);

	auto before = std::chrono::high_resolution_clock::now();

	cpu_dilate_binary(input.data(), cpu_output.data(), w, h, r, 255, timeout_us);

	auto after = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);

	return duration.count();
}

int time_us_nppi(const Case& cs)
{
	const int w = cs.width;
	const int h = cs.height;
	const float r = cs.radius;
	const auto& input = cs.array;

	size_t max_alloc = w * h * sizeof(uint8_t);
	uint8_t* d_input; cudaMalloc(&d_input, max_alloc);
	uint8_t* d_output; cudaMalloc(&d_output, max_alloc);
	void* d_workspace; cudaMalloc(&d_workspace, max_alloc);
	Npp8u* d_mask; cudaMalloc(&d_mask, max_alloc);

	cudaMemcpy(d_input, input.data(), max_alloc, cudaMemcpyHostToDevice);

	int k = int(r);
	std::vector<Npp8u> mask((2*k + 1) * (2*k + 1), 1);
	cudaMemcpy(d_mask, mask.data(), mask.size(), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();;
	auto before = std::chrono::high_resolution_clock::now();

	NppiSize oSizeROI  = { w, h };
	NppiPoint oOffset = { 0, 0 };
	NppiSize oMaskSize = { 2*k + 1, 2*k + 1 };
	NppiPoint oAnchor  = { k, k };
	NppStreamContext ctx{};
	cudaDeviceProp prop{};
	cudaGetDevice(&ctx.nCudaDeviceId);
	cudaGetDeviceProperties(&prop, ctx.nCudaDeviceId);
	ctx.hStream = nullptr;  // Default CUDA stream
	ctx.nMultiProcessorCount = prop.multiProcessorCount;
	ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
	ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
	ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
	ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
	ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
	cudaStreamGetFlags(ctx.hStream, &ctx.nStreamFlags);

	NppStatus status = nppiErodeBorder_8u_C1R_Ctx(
			d_input,
			w,
			oSizeROI,
			oOffset,
			d_output,
			w,
			oSizeROI,
			d_mask,
			oMaskSize,
			oAnchor,
			NPP_BORDER_REPLICATE,
			ctx
	);

	cudaDeviceSynchronize();;
	auto after = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(after - before);

	cudaFree(d_workspace);
	cudaFree(d_output);
	cudaFree(d_input);

	return 	(status == NPP_SUCCESS) ? duration.count() : -1;
}


int main(int argc, char** argv)
{
	std::cout << std::right;
	std::cout << std::setw(10) << "" << std::setw(10) << "this" << std::setw(10) << "nppi" << std::setw(10) << "cpu" << std::endl;
	std::cout << std::right;
	for (auto& csfunc : scale_cases()) {
		auto cs = csfunc();
		auto t1 = time_us_thislibrary(cs);
		auto t2 = time_us_nppi(cs);
		auto t3 = time_us_cpu_morphology(cs, 0);
		std::cout << std::setw(10) << cs.name << std::setw(10) << t1 << std::setw(10) << t2 << std::setw(10) << t3 << std::endl;
	}
	for (auto& csfunc : scale_cases_2()) {
		auto cs = csfunc();
		auto t1 = time_us_thislibrary(cs);
		auto t3 = time_us_cpu_morphology(cs, 0);
		std::cout << std::setw(10) << cs.name << std::setw(10) << t1 << std::setw(10) << "-" << std::setw(10) << t3 << std::endl;
	}
}
