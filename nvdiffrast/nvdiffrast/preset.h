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
	static void draw(GLbuffer& rb, GLint internalformat, GLenum format, float texminX, float texminY, float texmaxX, float texmaxY, float minX, float minY, float maxX, float maxY);
};

class PresetPrimitives {
	Matrix mat;

	Attribute target_pos;
	Attribute target_texel;
	Attribute target_normal;
	Attribute target_m_pos;
	Attribute target_r_normal;
	Texture target_diff;
	Texture target_rough;
	Texture target_nor;
	Texture target_disp;
	Attribute predict_pos;
	Attribute predict_texel;
	Attribute predict_normal;
	Attribute predict_m_pos;
	Attribute predict_r_normal;
	TextureGrad predict_diff;
	TextureGrad predict_rough;
	TextureGrad predict_nor;
	TextureGrad predict_disp;

	NormalcalcParams target_norm;
	ProjectParams target_proj;
	RasterizeParams target_rast;
	InterpolateParams target_intr;
	ProjectParams target_pos_proj;
	ProjectParams target_normal_proj;
	TexturemapParams target_diff_tex;
	TexturemapParams target_rough_tex;
	TexturemapParams target_nor_tex;
	TexturemapParams target_disp_tex;
	MaterialParams target_mtr;

	NormalcalcParams predict_norm;
	ProjectParams predict_proj;
	RasterizeParams predict_rast;
	InterpolateParams predict_intr;
	ProjectParams predict_pos_proj;
	ProjectParams predict_normal_proj;
	TexturemapGradParams predict_diff_tex;
	TexturemapGradParams predict_rough_tex;
	TexturemapGradParams predict_nor_tex;
	TexturemapGradParams predict_disp_tex;
	MaterialGradParams predict_mtr;

	GLbuffer target_diff_buffer;
	GLbuffer target_rough_buffer;
	GLbuffer target_nor_buffer;
	GLbuffer target_mtr_buffer;
	GLbuffer predict_diff_buffer;
	GLbuffer predict_rough_buffer;
	GLbuffer predict_nor_buffer;
	GLbuffer predict_mtr_buffer;

	LossParams loss;
	AdamParams diff_adam;
	AdamParams rough_adam;
	AdamParams nor_adam;

public:
	const int windowWidth = 2048;
	const int windowHeight = 1024;
	double t;
	void init();
	void display(void);
	void update(double dt);
	float getLoss() { return Loss::loss(loss);};
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
	const int windowWidth = 2048;
	const int windowHeight = 1024;
	void init(int resolution);
	void display();
	void update();
	float getLoss() { return Loss::loss(loss); };
};

class PresetEarth {
	Matrix mat;
	Attribute pos;
	Attribute texel;
	ProjectParams proj;
	RasterizeParams rast;
	InterpolateParams intr;
	struct Pass {
		Texture target_texture;
		TextureGrad predict_texture;

		TexturemapParams target_tex;
		TexturemapGradParams predict_tex;

		LossParams loss;
		AdamParams adam;
		void init(RasterizeParams& rast,InterpolateParams& intr ,int miplevel);
		void forward();
	};

	Pass mip;
	Pass nomip;

	GLbuffer gl_tex_predict;
	GLbuffer gl_predict;
	GLbuffer gl_tex_mip_predict;
	GLbuffer gl_mip_predict;
	GLbuffer gl_tex_target;
	GLbuffer gl_target;

public:
	const int windowWidth = 1536;
	const int windowHeight = 1024;
	void init();
	void display();
	void update();
	float getLoss() { return Loss::loss(mip.loss); };
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
	FilterParams target_flt;
	GLbuffer gl_target;

	ProjectGradParams predict_proj;
	RasterizeGradParams predict_rast;
	InterpolateGradParams predict_intr;
	AntialiasGradParams predict_aa;
	GLbuffer gl_aa_predict;
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

class PresetPhong {
	Matrix mat;

	Attribute pos;
	Attribute texel;
	Attribute normal;
	Attribute m_pos;
	Attribute r_normal;
	Texture target_texture;
	TextureGrad predict_texture;

	NormalcalcParams norm;
	ProjectParams proj;
	RasterizeParams rast;
	InterpolateParams intr;
	ProjectParams pos_proj;
	ProjectParams normal_proj;
	TexturemapParams target_tex;
	MaterialParams target_mtr;
	AntialiasParams target_aa;
	TexturemapGradParams predict_tex;
	MaterialGradParams predict_mtr;
	AntialiasGradParams predict_aa;

	LossParams loss;
	AdamParams mtr_adam;
	AdamParams tex_adam;

	GLbuffer predict_buffer;
	GLbuffer target_buffer;

public:
	const int windowWidth = 1024;
	const int windowHeight = 512;
	void init();
	void display(void);
	void update(void);
	float getLoss() { return Loss::loss(loss); };
};