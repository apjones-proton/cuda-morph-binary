#include "morphology.h"
#include  "macros.h"


namespace morph {

int kernel_size(float radius) {
	int n = int(radius);
	return 2 * n + 1;
}

size_t buffer_size(int width, int height, float radius) {
	int w64 = (width + 63) / 64;
	int area64 = w64 * height;
	int nkernel = kernel_size(radius);
	return 2 * area64 * sizeof(uint64_t) + nkernel * sizeof(int);
}

static __global__ void g_fill_circular_kernel(int* kernel, float radius) {
	int n = (int) radius;
	int nkernel = 2 * n + 1;
	FORALL(i, nkernel) {
		int y = i - n;
		int x = int(sqrtf(radius * radius - y * y));
		kernel[i] = x;
	}
}

static __global__ void g_compress_pixels_aligned(uint8_t* output, int area8, int w8, int w, const uint8_t* input)
{
	FORALL(i, area8) {
		int x = 8 * (i % w8), y = i / w8;
		int j = x + y * w;
		uint64_t mask = x < w ? *reinterpret_cast<const uint64_t*>(&input[j]) : 0;
		mask |= mask >> 4; // for each byte, compress any non-zero value from all 8 bits
		mask |= mask >> 2;
		mask |= mask >> 1;
		mask &= 0x0101010101010101;
		mask |= mask >> 28; // compress 8 bytes with zero or one to 8 bits
		mask |= mask >> 14;
		mask |= mask >> 7;
		mask &= 0xFF;
		output[i] = uint8_t(mask);
	}
}
static __global__ void g_compress_pixels_misaligned(uint8_t* output, int area8, int w8, int w, const uint8_t* input)
{
	FORALL(i, area8) {
		int x = 8 * (i % w8), y = i / w8;
		int j = x + y * w;
		uint8_t bits = 0;
		for (int b = 0; b < 8 && x < w; ++b, ++x, ++j) {
			uint8_t bytej = input[j];
			bytej |= bytej >> 4; // for each byte, compress any non-zero value from all 8 bits
			bytej |= bytej >> 2;
			bytej |= bytej >> 1;
			bits |= (bytej & 1) << b;
		}
		output[i] = bits;
	}
}

void compress_pixels(uint64_t* output, int area64, int w64, int w, const uint8_t* input)
{
	int area8 = area64 * 8, w8 = w64 * 8;
	if (w % 8 == 0)
		g_compress_pixels_aligned<<<GPU(area8)>>>(reinterpret_cast<uint8_t*>(output), area8, w8, w, input);
	else
		g_compress_pixels_misaligned<<<GPU(area8)>>>(reinterpret_cast<uint8_t*>(output), area8, w8, w, input);
}

static __global__ void g_decompress_pixels_aligned(uint8_t* output, int w, int area8, int w8, const uint8_t* input, uint8_t value)
{
	FORALL(i, area8) {
		int x = 8 * (i % w8), y = i / w8;
		int j = x + y * w;

		uint64_t mask = input[i];
		mask |= mask << 28; // expand 8 bits into 8 bytes with zero or one
		mask |= mask << 14;
		mask |= mask << 7;
		mask &= 0x0101010101010101;
		mask *= value; // zero or one becomes zero or value

		*reinterpret_cast<uint64_t*>(&output[j]) = mask;
	}
}
static __global__ void g_decompress_pixels_misaligned(uint8_t* output, int w, int area8, int w8, const uint8_t* input, uint8_t value)
{
	FORALL(i, area8) {
		int x = 8 * (i % w8), y = i / w8;
		int j = x + y * w;

		uint8_t bits = input[i];
		for (int b = 0; b < 8 && x < w; ++b, ++x, ++j) {
			output[j] = ((bits >> b) & 1) * value;
		}
	}
}
void decompress_pixels(uint8_t* output, int w, int area64, int w64, const uint64_t* input, uint8_t value)
{
	int area8 = area64 * 8, w8 = w64 * 8;
	if (w % 8 == 0)
		g_decompress_pixels_aligned<<<GPU(area8)>>>(output, w, area8, w8, reinterpret_cast<const uint8_t*>(input), value);
	else
		g_decompress_pixels_misaligned<<<GPU(area8)>>>(output, w, area8, w8, reinterpret_cast<const uint8_t*>(input), value);
}
__global__ void g_dilate_compressed(uint64_t* cmprs_out, int area64, int w64, int w, int h, const uint64_t* cmprs_in,
									int nkernel, const int* kernel)
{
	FORALL(i, area64) {
		int x0 = i % w64;
		int y0 = i / w64;
		uint64_t orig = cmprs_in[i];
		if (x0 == w64-1 && w < 64*w64) {
			int dw = w - 64*x0;
			orig &= (1<<dw - 1);
		}
		if (orig == 0) continue;
		int dymax = nkernel / 2; // assuming nkernel = 1 + 2 * d
		int y    = max(y0 - dymax, 0);
		int ymax = min(y0 + dymax, h-1);
		for ( ; y <= ymax; ++y) {
			int k = kernel[dymax + y - y0];
			uint64_t lft = 0, ctr = orig, rgt = 0;
			int dxlft = 0, dxrgt = 0;
			if (k < 64) {
				for (int w = 1; w < 2*k+1; ) {
					int d = min(w, k - w/2); w += 2*d;
					int xd = 64 - d;
					// confusingly <</>> means shift pixels to right/left
					auto old = ctr;
					ctr |= lft >> xd | old << d; lft |= lft >> d | old << xd;
					ctr |= old >> d | rgt << xd; rgt |= old >> xd | rgt << d;
				}
			} else {
				int lftpos = k - (__ffsll(orig) - 1);
				int rgtpos = k - __clzll(orig);
				lft = __brevll((uint64_t(1) << (lftpos % 64)) - 1);
				ctr = ~0;
				rgt = ((uint64_t(1) << (rgtpos % 64)) - 1);
				dxlft = lftpos / 64;
				dxrgt = rgtpos / 64;
			}
			{ int x = x0 - dxlft - 1; if (x >= 0 && lft) atomicOr(&cmprs_out[x + y*w64], lft); }
			for (int x =  x0 - dxlft; x <= x0 + dxrgt; ++x) if (x >= 0 && x < w64) {
				if (~ctr == 0) cmprs_out[x + y*w64] = ctr;
				else if (ctr) atomicOr(&cmprs_out[x + y*w64], ctr);
			}
			{ int x = x0 + dxrgt + 1; if (x < w64 && rgt) atomicOr(&cmprs_out[x + y*w64], rgt); }
		}
	}
}
__global__ void g_erode_compressed(uint64_t* cmprs_out, int area64, int w64, int w, int h, const uint64_t* cmprs_in,
								   int nkernel, const int* kernel)
{
	FORALL(i, area64) {
		int x0 = i % w64;
		int y0 = i / w64;
		uint64_t orig = cmprs_in[i];
		if (x0 == w64-1 && w < 64*w64) {
			int dw = w - 64*x0;
			orig |= ~(1<<dw - 1);
		}
		if (~orig == 0) continue;
		int dymax = nkernel / 2; // assuming nkernel = 1 + 2 * d
		int y    = max(y0 - dymax, 0);
		int ymax = min(y0 + dymax, h-1);
		for ( ; y <= ymax; ++y) {
			int k = kernel[dymax + y - y0];
			uint64_t lft = ~0, ctr = orig, rgt = ~0;
			int dxlft = 0, dxrgt = 0;
			if (k <= 64) {
				for (int w = 1; w < 2*k+1; ) {
					int d = min(w, k - w/2); w += 2*d;
					int xd = 64 - d;
					// confusingly <</>> means shift pixels to right/left
					auto old = ctr;
					ctr &= lft >> xd & old << d; lft &= lft >> d & old << xd;
					ctr &= old >> d & rgt << xd; rgt &= old >> xd & rgt << d;
				}
			} else {
				int lftpos = k - (__ffsll(~orig) - 1);
				int rgtpos = k - __clzll(~orig);
				lft = ~__brevll((uint64_t(1) << (lftpos % 64)) - 1);
				ctr = 0;
				rgt = ~((uint64_t(1) << (rgtpos % 64)) - 1);
				dxlft = lftpos / 64;
				dxrgt = rgtpos / 64;
			}
			// note if(~expr) is a short way of expressing 'if not all bits set in expr'
			{ int x = x0 - dxlft - 1; if (x >= 0 && ~lft) atomicAnd(&cmprs_out[x + y*w64], lft); }
			for (int x =  x0 - dxlft; x <= x0 + dxrgt; ++x) if (x >= 0 && x < w64) {
				if (ctr == 0) cmprs_out[x + y*w64] = ctr;
				else if (~ctr) atomicAnd(&cmprs_out[x + y*w64], ctr);
			}
			{ int x = x0 + dxrgt + 1; if (x < w64 && ~rgt) atomicAnd(&cmprs_out[x + y*w64], rgt); }
		}
	}
}

void dilate_binary(const uint8_t* input, uint8_t* output, int w, int h, float radius, void* workspace, uint8_t value)
{
	const int w64 = (w + 63) / 64;
	const int area64 = w64 * h;
	bool allocated = false;
	if (workspace == nullptr) { cudaMalloc(&workspace, buffer_size(w, h, radius)); allocated = true; }
	uint64_t* cmprs_in = reinterpret_cast<uint64_t*>(workspace);
	uint64_t* cmprs_out = reinterpret_cast<uint64_t*>(cmprs_in + area64);
	int* kernel = reinterpret_cast<int*>(cmprs_out + area64);
	int nkernel = kernel_size(radius);
	g_fill_circular_kernel<<<GPU(nkernel)>>>(kernel, radius);
	compress_pixels(cmprs_in, area64, w64, w, input);
	// dilate mask 64 bits at a time
	cudaMemset(cmprs_out, area64 * sizeof(uint64_t), 0);
	g_dilate_compressed<<<GPU(area64)>>>(cmprs_out, area64, w64, w, h, cmprs_in, nkernel, kernel);
	decompress_pixels(output, w, area64, w64, cmprs_out, value);
	if (allocated) { cudaFree(workspace); workspace = nullptr; }
}

void erode_binary(const uint8_t* input, uint8_t* output, int w, int h, float radius, void* workspace, uint8_t value)
{
	const int w64 = (w + 63) / 64;
	const int area64 = w64 * h;
	bool allocated = false;
	if (workspace == nullptr) { cudaMalloc(&workspace, buffer_size(w, h, radius)); allocated = true; }
	uint64_t* cmprs_in = reinterpret_cast<uint64_t*>(workspace);
	uint64_t* cmprs_out = reinterpret_cast<uint64_t*>(cmprs_in + area64);
	int* kernel = reinterpret_cast<int*>(cmprs_out + area64);
	int nkernel = kernel_size(radius);
	g_fill_circular_kernel<<<GPU(nkernel)>>>(kernel, radius);
	compress_pixels(cmprs_in, area64, w64, w, input);
	// erode mask 64 bits at a time
	cudaMemset(cmprs_out, area64 * sizeof(uint64_t), 0xFF);
	g_erode_compressed<<<GPU(area64)>>>(cmprs_out, area64, w64, w, h, cmprs_in, nkernel, kernel);
	decompress_pixels(output, w, area64, w64, cmprs_out, value);
	if (allocated) { cudaFree(workspace); workspace = nullptr; }
}

void close_binary(const uint8_t* input, uint8_t* output, int w, int h, float radius, void* workspace, uint8_t value)
{
	const int w64 = (w + 63) / 64;
	const int area64 = w64 * h;
	bool allocated = false;
	if (workspace == nullptr) { cudaMalloc(&workspace, buffer_size(w, h, radius)); allocated = true; }
	uint64_t* cmprs_1 = reinterpret_cast<uint64_t*>(workspace);
	uint64_t* cmprs_2 = reinterpret_cast<uint64_t*>(cmprs_1 + area64);
	int* kernel = reinterpret_cast<int*>(cmprs_2 + area64);
	int nkernel = kernel_size(radius);
	g_fill_circular_kernel<<<GPU(nkernel)>>>(kernel, radius);
	compress_pixels(cmprs_1, area64, w64, w, input);
	// dilate mask 64 bits at a time
	cudaMemset(cmprs_2, area64 * sizeof(uint64_t), 0);
	g_dilate_compressed<<<GPU(area64)>>>(cmprs_2, area64, w64, w, h, cmprs_1, nkernel, kernel);
	// erode mask 64 bits at a time
	cudaMemset(cmprs_1, area64 * sizeof(uint64_t), 0xFF);
	g_erode_compressed<<<GPU(area64)>>>(cmprs_1, area64, w64, w, h, cmprs_2, nkernel, kernel);
	// decompress mask 8 bits at a time (expand 8 bits to 64 bits padded with zeros)
	decompress_pixels(output, w, area64, w64, cmprs_1, value);
	if (allocated) { cudaFree(workspace); workspace = nullptr; }
}

void open_binary(const uint8_t* input, uint8_t* output, int w, int h, float radius, void* workspace, uint8_t value)
{
	const int w64 = (w + 63) / 64;
	const int area64 = w64 * h;
	bool allocated = false;
	if (workspace == nullptr) { cudaMalloc(&workspace, buffer_size(w, h, radius)); allocated = true; }
	uint64_t* cmprs_1 = reinterpret_cast<uint64_t*>(workspace);
	uint64_t* cmprs_2 = reinterpret_cast<uint64_t*>(cmprs_1 + area64);
	int* kernel = reinterpret_cast<int*>(cmprs_2 + area64);
	int nkernel = kernel_size(radius);
	g_fill_circular_kernel<<<GPU(nkernel)>>>(kernel, radius);
	compress_pixels(cmprs_1, area64, w64, w, input);
	// erode mask 64 bits at a time
	cudaMemset(cmprs_2, area64 * sizeof(uint64_t), 0xFF);
	g_erode_compressed<<<GPU(area64)>>>(cmprs_2, area64, w64, w, h, cmprs_1, nkernel, kernel);
	// dilate mask 64 bits at a time
	cudaMemset(cmprs_1, area64 * sizeof(uint64_t), 0);
	g_dilate_compressed<<<GPU(area64)>>>(cmprs_1, area64, w64, w, h, cmprs_2, nkernel, kernel);
	// decompress mask 8 bits at a time (expand 8 bits to 64 bits padded with zeros)
	decompress_pixels(output, w, area64, w64, cmprs_1, value);
	if (allocated) { cudaFree(workspace); workspace = nullptr; }
}

} // end namespace morph
