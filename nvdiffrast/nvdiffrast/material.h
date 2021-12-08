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
	float* texel;
	unsigned int* posidx;
	unsigned int* normalidx;
	unsigned int* texelidx;
	float* rast;
	float* diffusemap;
	float* roughnessmap;
	float* normalmap;
	float* heightmap;

	float* out;

	float3* eye;
	int lightNum;
	float3* point;
	float* lightintensity;
	float* params;
};

struct MaterialKernelGradParams {
	float* out;

	float* pos;
	float* normal;
	float* texel;
	float* diffusemap;
	float* roughnessmap;
	float* normalmap;
	float* heightmap;
	float3* point;
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
	static void init(MaterialParams& mtr, RasterizeParams& rast, ProjectParams& pos, ProjectParams& normal, Attribute* texel, int channel, float* diffusemap, float* roughnessmap, float* normalmap, float* heightmap);
	static void init(MaterialGradParams& mtr, RasterizeParams& rast, ProjectParams& pos, ProjectParams& normal, Attribute& texel, int channel, float* in, float* grad);
	static void init(MaterialParams& mtr, float3* eye, int lightNum, float3* point, float3* lightintensity);
};



class PhongMaterial :Material {
public:
	static void init(MaterialParams& mtr, float3* eye, int lightNum, float3* point, float3* lightintensity, float Ka, float Kd, float Ks, float shininess);
	static void forward(MaterialParams& mtr);
	static void forward(MaterialGradParams& mtr);
	static void backward(MaterialGradParams& mtr);
};



class PBRMaterial :Material {
public:
	static void init(MaterialParams& mtr, float3* eye, int lightNum, float3* point, float3* lightintensity, float ior);
	static void forward(MaterialParams& mtr);
	static void forward(MaterialGradParams& mtr);
	static void backward(MaterialGradParams& mtr);
};

