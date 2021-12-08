#pragma once
#include "common.h"
#include "buffer.h"
#include "matrix.h"
#include "project.h"
#include "rasterize.h"
#include "interpolate.h"
#include "texturemap.h"
#include "material.h"
#include "antialias.h"
#include "filter.h"
#include "normalcalc.h"
#include "loss.h"
#include "optimize.h"

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
	dim3 grid;
	dim3 block;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * kernel.channel * sizeof(float); };
};

class Noise {
public:
	static void init(NoiseParams& np, RasterizeParams& rast, float* in, int channel, float intensity);
	static void forward(NoiseParams& np);
};

struct GLbuffer {
	GLuint id;
	float* gl_buffer;
	float* buffer;
	int width;
	int height;
	int channel;
	size_t Size() { return (size_t)width * height * channel * sizeof(float); };
	static void init(GLbuffer& rb, float* buffer, int width, int height, int channel, int attachmentNum);
	static void draw(GLbuffer& rb, GLint internalformat, GLenum format, float minX, float minY, float maxX, float maxY);
};

class PresetPrimitives {
	Matrix mat;

	Attribute pos;
	Attribute texel;
	Attribute normal;
	Attribute m_pos;
	Attribute r_normal;
	Texture diff;
	Texture rough;
	Texture nor;
	Texture disp;

	NormalcalcParams norm;
	ProjectParams proj;
	RasterizeParams rast;
	InterpolateParams intr;
	ProjectParams pos_proj;
	ProjectParams normal_proj;
	TexturemapParams diff_tex;
	TexturemapParams rough_tex;
	TexturemapParams nor_tex;
	TexturemapParams disp_tex;
	MaterialParams mtr;
	AntialiasParams aa;
	FilterParams flt;
	GLbuffer rast_buffer;
	GLbuffer intr_buffer;
	GLbuffer diff_buffer;
	GLbuffer mtr_buffer;
	GLbuffer aa_buffer;
	GLbuffer flt_buffer;

public:
	const int windowWidth = 1536;
	const int windowHeight = 1024;
	void init();
	void display(void);
	void update(void);
	float getLoss() { return 0.0; };
};

class PresetCube {
	Matrix mat;
	Matrix hr_mat;

	Attribute target_pos;
	Attribute target_color;
	AttributeGrad predict_pos;
	AttributeGrad predict_color;

	ProjectParams target_proj;
	RasterizeParams target_rast;
	InterpolateParams target_intr;
	AntialiasParams target_aa;
	GLbuffer gl_target;

	ProjectGradParams predict_proj;
	RasterizeGradParams predict_rast;
	InterpolateGradParams predict_intr;
	AntialiasGradParams predict_aa;
	GLbuffer gl_predict;

	ProjectParams hr_target_proj;
	RasterizeParams hr_target_rast;
	InterpolateParams hr_target_intr;
	AntialiasParams hr_target_aa;
	GLbuffer gl_hr_target;

	ProjectParams hr_predict_proj;
	RasterizeParams hr_predict_rast;
	InterpolateParams hr_predict_intr;
	AntialiasParams hr_predict_aa;
	GLbuffer gl_hr_predict;

	LossParams loss;
	AdamParams adam_pos;
	AdamParams adam_color;

public:
	const int windowWidth = 1024;
	const int windowHeight = 1024;
	void init(int resolution);
	void display();
	void update();
	float getLoss() { return Loss::loss(loss); };
};

class PresetEarth {
	int batch;
	Matrix mat;
	Attribute pos;
	Attribute texel;
	Texture target_texture;
	GLbuffer gl_tex_target;
	TextureGrad predict_texture;
	GLbuffer gl_tex_predict;

	ProjectParams proj;
	RasterizeParams rast;
	InterpolateParams intr;
	TexturemapParams target_tex;
	AntialiasParams target_aa;
	GLbuffer gl_target;
	TexturemapGradParams predict_tex;
	AntialiasGradParams predict_aa;
	GLbuffer gl_predict;

	LossParams loss;
	AdamParams adam;

public:
	const int windowWidth = 2560;
	const int windowHeight = 1024;
	void init();
	void display();
	void update();
	float getLoss() { return Loss::loss(loss); };
};

//original preset

class PresetFilter {
	Matrix mat;
	Matrix hr_mat;

	Attribute target_pos;
	Attribute target_color;
	AttributeGrad predict_pos;
	AttributeGrad predict_color;

	ProjectParams target_proj;
	RasterizeParams target_rast;
	InterpolateParams target_intr;
	AntialiasParams target_aa;
	GLbuffer gl_aa_target;
	NoiseParams target_np;
	GLbuffer gl_np_target;
	FilterParams target_flt;
	GLbuffer gl_target;

	ProjectGradParams predict_proj;
	RasterizeGradParams predict_rast;
	InterpolateGradParams predict_intr;
	AntialiasGradParams predict_aa;
	FilterGradParams predict_flt;
	GLbuffer gl_predict;

	ProjectParams hr_target_proj;
	RasterizeParams hr_target_rast;
	InterpolateParams hr_target_intr;
	AntialiasParams hr_target_aa;
	GLbuffer gl_hr_target;

	ProjectParams hr_predict_proj;
	RasterizeParams hr_predict_rast;
	InterpolateParams hr_predict_intr;
	AntialiasParams hr_predict_aa;
	GLbuffer gl_hr_predict;

	LossParams loss;
	AdamParams adam_pos;
	AdamParams adam_color;

public:
	const int windowWidth = 1536;
	const int windowHeight = 1024;
	void init(int resolution,int k);
	void display();
	void update();
	float getLoss() { return Loss::loss(loss); };
};