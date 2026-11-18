#pragma once
#include <cstddef>
#include <cstdint>

namespace morph {

size_t buffer_size(int width, int height, float radius);

void dilate_binary(const uint8_t* input, uint8_t* output, int width, int height, float radius, void* buffer = nullptr, uint8_t on = 255);
void erode_binary(const uint8_t* input, uint8_t* output, int width, int height, float radius, void* buffer = nullptr, uint8_t on = 255);
void close_binary(const uint8_t* input, uint8_t* output, int width, int height, float radius, void* buffer = nullptr, uint8_t on = 255);
void open_binary(const uint8_t* input, uint8_t* output, int width, int height, float radius, void* buffer = nullptr, uint8_t on = 255);

}
