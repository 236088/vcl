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



struct SGSpecularKernelParams{
	int width;
	int height;
	int depth;

	float* rast;

	float* normal;
	float* view;

	float* roughness;
	float ior;

	float* outAxis;
	float* outSharpness;
	float* outAmplitude;
};

struct SGSpecularGradKernelParams{
	float* normal;
	float* view;

	float* roughness;
	float ior;

	float* out;
};

struct SGSpecularParams {
	SGSpecularKernelParams kernel;
	size_t Size(int dimention) { return (size_t)kernel.width * kernel.height * kernel.depth * dimention * sizeof(float); };
};

struct SGSpecularGradParams :SGSpecularParams {
	SGSpecularGradKernelParams grad;
};

class SGSpecular {
public:
	static void init(SGSpecularParams& spec, RasterizeParams& rast, NormalAxisParams& normal, ViewAxisParams& view, TexturemapParams& roughness, float ior);
	static void forward(SGSpecularParams& spec);
};



struct SphericalGaussianKernelParams {
	int width;
	int height;
	int depth;
	int channel;

	float* rast;

	int sgnum;
	float* axis;
	float* sharpness;
	float* amplitude;

	float* normal;
	float* view;
	float* diffuse;

	float* specAxis;
	float* specSharpness;
	float* specAmplitude;

	float* out;
	float* outDiffenv;
	float* outSpecenv;
};

struct SphericalGaussianGradKernelParams {
	float* normal;
	float* view;
	float* diffuse;

	float* axis;
	float* sharpness;
	float* amplitude;

	float* specAxis;
	float* specSharpness;
	float* specAmplitude;

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
	static void init(SphericalGaussianParams& sg, RasterizeParams& rast, NormalAxisParams& normal, ViewAxisParams& view, TexturemapParams& diffuse, SGSpecularParams& spec, SGBuffer& sgbuf);
	static void init(SphericalGaussianGradParams& sg, RasterizeParams& rast, NormalAxisParams& normal, ViewAxisParams& view, TexturemapParams& diffuse, SGSpecularParams& spec, SGBufferGrad& sgbuf);
	static void init(SphericalGaussianGradParams& sg, RasterizeParams& rast, NormalAxisParams& normal, ViewAxisParams& view, TexturemapGradParams& diffuse, SGSpecularParams& spec, SGBufferGrad& sgbuf);
	static void forward(SphericalGaussianParams& sg);
	static void forward(SphericalGaussianGradParams& sg);
	static void backward(SphericalGaussianGradParams& sg);
};