#pragma once
#include "common.h"
#include "buffer.h"
#include "rasterize.h"

struct NoiseKernelParams {
	int width;
	int height;
	int depth;
	int channel;
	float intensity;
	float* in;

	float* out;
};

struct NoiseParams {
	NoiseKernelParams kernel;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * kernel.channel * sizeof(float); };
};

class Noise {
public:
	static void init(NoiseParams& np, RasterizeParams& rast, float* in, int channel, float intensity);
	static void forward(NoiseParams& np);
};

