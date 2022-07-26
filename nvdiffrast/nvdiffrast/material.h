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
	float* in;

	float* out;

	float3 eye;
	int lightNum;
	float* point;
	float* intensity;
	float* params;
};

struct MaterialKernelGradParams {
	float* out;

	float* pos;
	float* normal;
	float* texel;
	float* point;
	float* intensity;
	float* params;
	float* in;
};

struct MaterialParams {
	MaterialKernelParams kernel;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * kernel.channel * sizeof(float); };
};

struct MaterialGradParams : MaterialParams {
	MaterialKernelGradParams grad;
};

class Material {
public:
	static void init(MaterialParams& mtr, RasterizeParams& rast, ProjectParams& pos, ProjectParams& normal, Attribute* texel, int channel, float* in);
	static void init(MaterialParams& mtr, float3 eye, Buffer& point, Buffer& intensity);
	static void init(MaterialParams& mtr, Buffer& params);
	static void init(MaterialGradParams& mtr, float3 eye, BufferGrad& point, BufferGrad& intensity);
	static void init(MaterialGradParams& mtr, BufferGrad& params);
	static void init(MaterialGradParams& mtr, RasterizeParams& rast, ProjectParams& pos, ProjectParams& normal, Attribute* texel, int channel, float* in, float* grad);
};



class PhongMaterial :Material {
public:
	static void forward(MaterialParams& mtr);
	static void forward(MaterialGradParams& mtr);
	static void backward(MaterialGradParams& mtr);
};