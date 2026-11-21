#include "cases.h"

Case empty(int w, int h, float r)
{
	Case cs;
	cs.width = w;
	cs.height = h;
	cs.radius = r;
	cs.array = std::vector<uint8_t>(w * h);
	return cs;
}

void set_pixel(Case& cs, int x, int y)
{
	cs.array[x + y * cs.width] = 255;
}

void set_block(Case& cs, int x0, int y0, int x1, int y1)
{
	for (int y = y0; y <= y1; ++y)
		for (int x = x0; x <= x1; ++x)
			set_pixel(cs, x, y);
}

void set_diamond(Case& cs, int x0, int y0, int size)
{
	int ymin = std::max(y0 - size, 0);
	int ymax = std::min(y0 + size, cs.height-1);
	for (int y = ymin; y <= ymax; ++y) {
		int xsize = size - std::abs(y - y0);
		int xmin = std::max(x0 - xsize, 0);
		int xmax = std::min(x0 + xsize, cs.width-1);
		for (int x = xmin; x <= xmax; ++x)
			set_pixel(cs, x, y);
	}
}

void set_circle(Case& cs, int x0, int y0, int radius)
{
	int ymin = std::max(y0 - radius, 0);
	int ymax = std::min(y0 + radius, cs.height-1);
	for (int y = ymin; y <= ymax; ++y) {
		int xmin = std::max(x0 - radius, 0);
		int xmax = std::min(x0 + radius, cs.width-1);
		for (int x = xmin; x <= xmax; ++x) {
			int dx = x - x0, dy = y - y0;
			if (dx * dx + dy * dy < radius * radius)
				set_pixel(cs, x, y);
		}
	}
}


Case case_1()
{
	auto cs = empty(4096, 4096, 100.0f);
	set_pixel(cs, 23,47);
	set_pixel(cs, 1021, 3024);
	set_pixel(cs, 512, 100);
	cs.name = __FUNCTION__;
	return cs;
}
Case case_2()
{
	auto cs = empty(1024, 1024, 10.0f);
	set_pixel(cs, 23,47);
	set_pixel(cs, 55, 66);
	set_pixel(cs, 90, 10);
	cs.name = __FUNCTION__;
	return cs;
}

Case case_3()
{
	auto cs = empty(3333, 3333, 47.0f);
	set_circle(cs, 3300, 15, 300);
	set_circle(cs, 30, 0, 300);
	set_circle(cs, 3300, 3200, 300);
	set_circle(cs, 45, 3200, 300);
	set_diamond(cs, 1000, 1000, 500);
	cs.name = __FUNCTION__;
	return cs;
}

Case case_4()
{
	auto cs = empty(3333, 3333, 47.0f);
	set_pixel(cs, 3300, 15);
	set_pixel(cs, 30, 0);
	set_pixel(cs, 3300, 3200);
	set_pixel(cs, 45, 3200);
	set_pixel(cs, 1000, 1000);
	cs.name = __FUNCTION__;
	return cs;
}

Case case_5()
{
	auto cs = empty(127, 127, 47.0f);
	set_pixel(cs, 30, 0);
	cs.name = __FUNCTION__;
	return cs;
}

std::vector<std::function<Case()>> all_cases()
{
	return { case_1, case_2, case_3, case_4, case_5 };
}
