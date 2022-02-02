#pragma once
#include "common.h"
#include "buffer.h"
#include "rasterize.h"

struct FilterKernelParams {
	int width;
	int height;
	int depth;
	int channel;
	int k;

	float* sigma;
	float* filter;
	float* in;
	float* buf;

	float* out;
};

struct FilterKernelGradParams {
	float* in;
	float* sigma;
	float* filter;
	float* buf;
	float* bufx;
	float* bufy;

	float* out;
};

struct FilterParams {
	float h_sig;
	FilterKernelParams kernel;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * kernel.channel * sizeof(float); };
};

struct FilterGradParams :FilterParams {
	FilterKernelGradParams grad;
};

class Filter {
public:
	static void init(FilterParams& flt, RasterizeParams& rast, float* in, int channel, float sigma);
	static void init(FilterGradParams& flt, RasterizeParams& rast, float* in, float* grad, int channel, float sigma);
	static void forward(FilterParams& flt);
	static void forward(FilterGradParams& flt);
	static void backward(FilterGradParams& flt);
};
