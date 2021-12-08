#pragma once
#include "common.h"
#include "buffer.h"
#include "rasterize.h"
#include "interpolate.h"

struct MaterialKernelParams {
	int width;
	int height;
	int depth;
	int channel;
	float* pos;
	float* normal;
	float* rast;
	float* in;

	float* out;

	float3* eye;
	int lightNum;
	float3* direction;
	float* lightintensity;
	float* params;
};

struct MaterialKernelGradParams {
	float* out;

	float* pos;
	float* normal;
	float* in;
	float3* direction;
	float3* lightintensity;
	float* params;
};

struct MaterialParams {
	MaterialKernelParams kernel;
	dim3 block;
	dim3 grid;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * kernel.channel * sizeof(float); };
};

struct MaterialGradParams : MaterialParams {
	MaterialKernelGradParams grad;
};

class Material {
public:
	static void init(MaterialParams& mtr, RasterizeParams& rast, InterpolateParams& pos, InterpolateParams& normal, int channel, float* in);
	static void init(MaterialGradParams& mtr, RasterizeParams& rast, InterpolateParams& pos, InterpolateParams& normal, int channel, float* in, float* grad);
	static void init(MaterialParams& mtr, float3* eye, int lightNum, float3* direction, float3* lightintensity);
};



class PhongMaterial :Material {
public:
	static void init(MaterialParams& mtr, float3* eye, int lightNum, float3* direction, float3* lightintensity, float Ka, float Kd, float Ks, float shininess);
	static void forward(MaterialParams& mtr);
	static void forward(MaterialGradParams& mtr);
	static void backward(MaterialGradParams& mtr);
};



class PBRMaterial :Material {
public:
	static void init(MaterialParams& mtr, float3* eye, int lightNum, float3* direction, float3* lightintensity, float roughness, float ior);
	static void forward(MaterialParams& mtr);
	static void forward(MaterialGradParams& mtr);
	static void backward(MaterialGradParams& mtr);
};

