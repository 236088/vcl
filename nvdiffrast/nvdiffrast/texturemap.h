#pragma once
#include "common.h"
#include "buffer.h"
#include "rasterize.h"
#include "interpolate.h"


struct TexturemapKernelParams {
	int width;
	int height;
	int depth;
	int texwidth;
	int texheight;
	int channel;
	int miplevel;

	float* rast;
	float* uv;
	float* uvDA;
	float* texture[TEX_MAX_MIP_LEVEL];

	float* out;
};

struct TexturemapKernelGradParams {
	float* out;

	float* uv;
	float* uvDA;
	float* texture[TEX_MAX_MIP_LEVEL];
};

struct TexturemapParams {
	TexturemapKernelParams kernel;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * kernel.channel * sizeof(float); };
};

struct TexturemapGradParams :TexturemapParams {
	TexturemapKernelGradParams grad;
};

class Texturemap {
public:
	static void init(TexturemapParams& tex, RasterizeParams& rast, InterpolateParams& intr, Texture& texture);
	static void init(TexturemapGradParams& tex, RasterizeParams& rast, InterpolateGradParams& intr, TextureGrad& texture);
	static void init(TexturemapGradParams& tex, RasterizeParams& rast, InterpolateParams& intr, TextureGrad& texture);
	static void init(TexturemapGradParams& tex, RasterizeParams& rast, InterpolateGradParams& intr, Texture& texture);
	static void forward(TexturemapParams& tex);
	static void forward(TexturemapGradParams& tex);
	static void backward(TexturemapGradParams& tex);
};