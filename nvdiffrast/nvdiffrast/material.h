#pragma once
#include "common.h"
#include "buffer.h"
#include "matrix.h"
#include "rasterize.h"
#include "interpolate.h"
#include "texturemap.h"

struct NormalAxisKernelParams {
	int width;
	int height;
	int depth;
	float* rast;

	float* rot;
	float* normal;
	unsigned int* normalidx;
	float* pos;
	unsigned int* posidx;
	float* texel;
	unsigned int* texelidx;
	float* normalmap;
	
	float* out;
};

struct NormalAxisGradKernelParams {
	float* normal;
	float* normalmap;

	float* out;
};

struct NormalAxisParams {
	NormalAxisKernelParams kernel;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * 3 * sizeof(float); };
};

struct NormalAxisGradParams : NormalAxisParams {
	NormalAxisGradKernelParams grad;
};

class NormalAxis {
public:
	static void init(NormalAxisParams& norm, RotationParams& rot, RasterizeParams& rast, Attribute& normal);
	static void init(NormalAxisParams& norm, RotationParams& rot, RasterizeParams& rast, Attribute& normal, Attribute& pos, Attribute& texel, TexturemapParams& normalmap);
	static void forward(NormalAxisParams& norm);
};



struct ViewAxisKernelParams{
	int width;
	int height;
	int depth;

	float* rast;

	float* rot;
	glm::mat4* view;
	glm::mat4* projection;
	float* pvinv;

	float* out;
};

struct ViewAxisGradKernelParams{
	float* rast;
	float* normal;

	glm::mat4* view;
	glm::mat4* projection;

	float* out;
};

struct ViewAxisParams {
	ViewAxisKernelParams kernel;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * 3 * sizeof(float); };
};

struct ViewAxisGradParams : ViewAxisParams {
	ViewAxisGradKernelParams grad;
};

class ViewAxis {
public:
	static void init(ViewAxisParams& view, RotationParams& rot, CameraParams& cam, RasterizeParams& rast);
	static void forward(ViewAxisParams& view);
};



struct SphericalGaussianKernelParams {
	int width;
	int height;
	int depth;
	int channel;

	float* rast;

	float* normal;
	float* view;
	float* diffuse;
	float* roughness;
	float ior;

	int sgnum;
	float* axis;
	float* sharpness;
	float* amplitude;

	float* out;
	float* outDiffenv;
	float* outSpecenv;
};

struct SphericalGaussianGradKernelParams {
	float* normal;
	float* view;
	float* diffuse;
	float* roughness;
	float ior;

	float* axis;
	float* sharpness;
	float* amplitude;

	float* out;
};

struct SphericalGaussianParams {
	SphericalGaussianKernelParams kernel;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * kernel.channel * sizeof(float); };
};

struct SphericalGaussianGradParams : SphericalGaussianParams{
	SphericalGaussianGradKernelParams grad;
};

class SphericalGaussian {
public:
	static void init(SphericalGaussianParams& sg, RasterizeParams& rast, NormalAxisParams& normal, ViewAxisParams& view, TexturemapParams& diffuse, TexturemapParams& roughness, SGBuffer& sgbuf, float ior);
	static void forward(SphericalGaussianParams& sg);
	static void backward(SphericalGaussianGradParams& sg);
};