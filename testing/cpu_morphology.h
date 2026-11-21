#pragma once
#include <cstddef>
#include <cstdint>

void cpu_dilate_binary(const uint8_t* input, uint8_t* output, int width, int height, float radius, uint8_t on = 255, int timeout_us = 0);
void cpu_erode_binary(const uint8_t* input, uint8_t* output, int width, int height, float radius, uint8_t on = 255, int timeout_us = 0);
void cpu_close_binary(const uint8_t* input, uint8_t* output, int width, int height, float radius, uint8_t on = 255, int timeout_us = 0);
void cpu_open_binary(const uint8_t* input, uint8_t* output, int width, int height, float radius, uint8_t on = 255, int timeout_us = 0);
