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

Case case_1();
Case case_2();
Case case_3();
Case case_4();

std::vector<std::function<Case()>> all_cases();
