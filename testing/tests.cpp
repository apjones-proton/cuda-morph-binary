#include "../include/morphology.h"
#include "cases.h"
#include "cpu_morphology.h"
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdio>

bool test_correctness(const Case& cs)
{
	const int w = cs.width;
	const int h = cs.height;
	const float r = cs.radius;
	const auto& input = cs.array;

	bool result = true;
	uint8_t* d_input; cudaMalloc(&d_input, w * h * sizeof(uint8_t));
	uint8_t* d_output; cudaMalloc(&d_output, w * h * sizeof(uint8_t));
	void* d_workspace; cudaMalloc(&d_workspace, morph::buffer_size(w, h, r));
	std::vector<uint8_t> output(w * h * sizeof(uint8_t));
	std::vector<uint8_t> cpu_output(w * h * sizeof(uint8_t));

	cudaMemcpy(d_input, input.data(), w * h * sizeof(uint8_t), cudaMemcpyHostToDevice);
	morph::dilate_binary(d_input, d_output, w, h, r, d_workspace);
	cudaMemcpy(output.data(), d_output, w * h * sizeof(uint8_t), cudaMemcpyDeviceToHost);
	cpu_dilate_binary(input.data(), cpu_output.data(), w, h, r);
	cv::Mat cpu_image(h, w, CV_8UC1, const_cast<uint8_t*>(cpu_output.data()));
	cv::Mat gpu_image(h, w, CV_8UC1, const_cast<uint8_t*>(output.data()));
	cv::imwrite("cpu.jpg", cpu_image);
	cv::imwrite("gpu.jpg", gpu_image);
	for (int i=0, y=0; y < h; ++y) {
		for (int x=0; x < w; ++x, ++i) {
			if (output[i] != cpu_output[i]) {
				printf("different at pixel (%d,%d) %d != %d\n", x, y, output[i], cpu_output[i]);
				result = false;
				goto breakbreak;
			}
		}
	}
	breakbreak:

	cudaFree(d_workspace);
	cudaFree(d_output);
	cudaFree(d_input);

	return result;
}

int main(int argc, char** argv)
{
	for (auto& cs_func : all_cases()) {
		Case cs = cs_func();
		std::cout << cs.name << " " << cs.width << " " << cs.height << " " << cs.radius << std::endl;
		test_correctness(cs);
	}
}
