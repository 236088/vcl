#pragma once
#include "common.h"
#include "buffer.h"
#include "rasterize.h"
#include "interpolate.h"

struct MaterialKernelParams {
	int width;
	int height;
	int depth;
	float* pos;
	float* normal;
	float* rast;
	float* in;

	float* out;

	float3* eye;
	int lightNum;
	float3* lightpos;
	float3* lightintensity;
	float* params;
};

struct MaterialKernelGradParams {
	float* out;

	float* pos;
	float* normal;
	float* in;
	float3* lightpos;
	float3* lightintensity;
	float* params;
};

struct MaterialParams {
	MaterialKernelParams kernel;
	MaterialKernelGradParams grad;
	dim3 block;
	dim3 grid;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * 3 * sizeof(float); };
};

class Material {
public:
	static void init(MaterialParams& mtr, RasterizeParams& rast, InterpolateParams& pos, InterpolateParams& normal, float* in);
	static void init(MaterialParams& mtr, float3* eye, int lightNum, float3* lightpos, float3* lightintensity, float3 ambient, float Ka, float Kd, float Ks, float shininess);
	static void init(MaterialParams& mtr, float* dLdout);
	static void clear(MaterialParams& mtr);
	static void forward(MaterialParams& mtr);
	static void backward(MaterialParams& mtr);
};

