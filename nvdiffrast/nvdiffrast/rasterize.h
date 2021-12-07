#pragma once
#include "common.h"
#include "buffer.h"
#include "project.h"

struct RasterizeKernelParams {
	int width;
	int height;
	int depth;
	int enableDB;
	float xs;
	float ys;
	float* proj;
	unsigned int* idx;

	float* out;
	float* outDB;
};

struct RasterizeKernelGradParams{
	float* proj;

	float* out;
	float* outDB;
};

struct RasterizeParams {
	RasterizeKernelParams kernel;
	dim3 grid;
	dim3 block;

	int projNum;
	int idxNum;

	GLuint fbo;
	GLuint program;
	GLuint buffer;
	GLuint bufferDB;

	float* gl_proj;
	unsigned int* gl_idx;
	float* gl_out;
	float* gl_outDB;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * 4 * sizeof(float); };
};

struct RasterizeGradParams : RasterizeParams {
	RasterizeKernelGradParams grad;
};

class Rasterize {
public:
	static void init(RasterizeParams& rast, ProjectParams& proj, int width, int height, int depth, bool enableDB);
	static void init(RasterizeGradParams& rast, ProjectGradParams& proj, int width, int height, int depth, bool enableDB);
	static void forward(RasterizeParams& rast);
	static void forward(RasterizeGradParams& rast);
	static void backward(RasterizeGradParams& rast);
};