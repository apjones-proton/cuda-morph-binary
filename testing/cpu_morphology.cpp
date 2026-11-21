#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>

void cpu_dilate_binary(const uint8_t* input, uint8_t* output, int width, int height, float radius, uint8_t on, int timeout_us)
{
	const int w = width;
	const int h = height;
	const int area = w * h;
	const int n = int(radius);
	for (int i = 0; i < area; ++i) output[i] = input[i] ? on : 0;
	auto start = std::chrono::high_resolution_clock::now();
	for (int i = 0, y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x, ++i) {
			if (input[i]) {
				if ((x==0 || input[i-1]) && (x==w-1 || input[i+1]) && (y==0 || input[i-w]) && (y==h-1 || input[i+w])) {
					continue;
				}
				auto now = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
				if (timeout_us && duration.count() > timeout_us) return;
				int minx = std::max(x - n, 0);
				int maxx = std::min(x + n, w - 1);
				int miny = std::max(y - n, 0);
				int maxy = std::min(y + n, h - 1);
				for (int y2 = miny; y2 <= maxy; ++y2) {
					int i2 = minx + y2 * width;
					for (int x2 = minx; x2 <= maxx; ++x2, ++i2) {
						int dx = x2-x, dy = y2-y;
						if (dx * dx + dy * dy <= radius * radius) output[i2] = on;
					}
				}
			}
		}
	}
}
