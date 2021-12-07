#pragma once
#include "common.h"
#include "buffer.h"
#include "rasterize.h"


struct AntialiasKernelParams {
	int width;
	int height;
	int depth;
	int channel;
	float xh;
	float yh;

	float* proj;
	unsigned int* idx;
	float* rast;
	float* in;

	float* out;
};

struct AntialiasKernelGradParams {
	float* proj;
	float* in;

	float* out;
};

struct AntialiasParams {
	AntialiasKernelParams kernel;
	dim3 block;
	dim3 grid;
	int projNum;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * kernel.channel * sizeof(float); };
};

struct AntialiasGradParams:AntialiasParams{
	AntialiasKernelGradParams grad;
};

class Antialias {
public:
	static void init(AntialiasParams& aa, RasterizeParams& rast, ProjectParams& proj, float* in, int channel);
	static void init(AntialiasGradParams& aa, RasterizeParams& rast, ProjectGradParams& proj, float* in, float* grad, int channel);
	static void init(AntialiasGradParams& aa, RasterizeParams& rast, ProjectGradParams& proj, float* in, int channel);
	static void init(AntialiasGradParams& aa, RasterizeParams& rast, ProjectParams& proj, float* in, float* grad, int channel);
	static void forward(AntialiasParams& aa);
	static void forward(AntialiasGradParams& aa);
	static void backward(AntialiasGradParams& aa);
};
