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

	float* filter;
	float* in;
	float* buf;

	float* out;
};

struct FilterKernelGradParams {
	float* in;

	float* out;
};

struct FilterParams {
	FilterKernelParams kernel;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * kernel.channel * sizeof(float); };
};

struct FilterGradParams :FilterParams {
	FilterKernelGradParams grad;
};

class Filter {
protected:
	static void init(FilterParams& flt, RasterizeParams& rast, float* in, int channel, int k);
	static void init(FilterGradParams& flt, RasterizeParams& rast, float* in, float* grad, int channel, int k);
public:
	static void forward(FilterParams& flt);
	static void forward(FilterGradParams& flt);
	static void backward(FilterGradParams& flt);
};

class GaussianFilter :Filter {
public:
	static void init(FilterParams& flt, RasterizeParams& rast, float* in, int channel, int k);
	static void init(FilterGradParams& flt, RasterizeParams& rast, float* in, float* grad, int channel, int k);
};

class MeanFilter:Filter {
public:
	static void init(FilterParams& flt, RasterizeParams& rast, float* in, int channel, int k);
	static void init(FilterGradParams& flt, RasterizeParams& rast, float* in, float* grad, int channel, int k);
};
