#pragma once

#include "common.h"
#include "buffer.h"
#include "rasterize.h"

struct InterpolateKernelParams {
	int width;
	int height;
	int depth;
	int enableDA;
	float* rast;
	float* rastDB;
	float* attr;
	unsigned int* idx;
	int dimention;

	float* out;
	float* outDA;
};

struct InterpolateKernelGradParams {
	float* out;
	float* outDA;

	float* rast;
	float* rastDB;
	float* attr;
};

struct InterpolateParams {
	InterpolateKernelParams kernel;
	dim3 grid;
	dim3 block;
	int attrNum;
	int idxNum;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * kernel.dimention * sizeof(float); };
};

struct InterpolateGradParams : InterpolateParams{
	InterpolateKernelGradParams grad;
};

class Interpolate {
public:
	static void init(InterpolateParams& intr, RasterizeParams& rast, Attribute& attr);
	static void init(InterpolateGradParams& intr, RasterizeGradParams& rast, AttributeGrad& attr);
	static void init(InterpolateGradParams& intr, RasterizeGradParams& rast, Attribute& attr);
	static void init(InterpolateGradParams& intr, RasterizeParams& rast, AttributeGrad& attr);
	static void forward(InterpolateParams& intr);
	static void forward(InterpolateGradParams& intr);
	static void backward(InterpolateGradParams& intr);
};