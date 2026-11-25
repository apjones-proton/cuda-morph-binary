#pragma once
#include <functional>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>


struct Case {
	int width;
	int height;
	float radius;
	std::vector<uint8_t> array;
	std::string name;
};

std::vector<std::function<Case()>> all_cases();

std::vector<std::function<Case()>> scale_cases();
std::vector<std::function<Case()>> scale_cases_2();
