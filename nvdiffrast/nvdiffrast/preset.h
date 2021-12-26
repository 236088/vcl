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
	static void init(GLbuffer& rb, float* buffer, int width, int height, int channel);
	static void draw(GLbuffer& rb, GLint internalformat, GLenum format, float minX, float minY, float maxX, float maxY);
	static void draw(GLbuffer& rb, GLint internalformat, GLenum format, float texminX, float texminY, float texmaxX, float texmaxY, float minX, float minY, float maxX, float maxY);
};

class PresetCube {
	Matrix mat;
	Matrix hr_mat;

	Attribute target_pos;
	Attribute target_color;

	ProjectParams target_proj;
	RasterizeParams target_rast;
	InterpolateParams target_intr;
	AntialiasParams target_aa;
	GLbuffer gl_target;

	ProjectParams hr_target_proj;
	RasterizeParams hr_target_rast;
	InterpolateParams hr_target_intr;
	AntialiasParams hr_target_aa;
	GLbuffer gl_hr_target;

	AttributeGrad predict_pos;
	AttributeGrad predict_color;

	ProjectGradParams predict_proj;
	RasterizeGradParams predict_rast;
	InterpolateGradParams predict_intr;
	AntialiasGradParams predict_aa;
	GLbuffer gl_predict;

	ProjectParams hr_predict_proj;
	RasterizeParams hr_predict_rast;
	InterpolateParams hr_predict_intr;
	AntialiasParams hr_predict_aa;
	GLbuffer gl_hr_predict;

	LossParams predict_loss;
	AdamParams predict_adam_pos;
	AdamParams predict_adam_color;

	AttributeGrad noaa_pos;
	AttributeGrad noaa_color;

	ProjectGradParams noaa_proj;
	RasterizeGradParams noaa_rast;
	InterpolateGradParams noaa_intr;
	GLbuffer gl_noaa;

	ProjectParams hr_noaa_proj;
	RasterizeParams hr_noaa_rast;
	InterpolateParams hr_noaa_intr;
	AntialiasParams hr_noaa_aa;
	GLbuffer gl_hr_noaa;

	LossParams noaa_loss;
	AdamParams noaa_adam_pos;
	AdamParams noaa_adam_color;

	float predict_loss_sum;
	float noaa_loss_sum;
	int step;
	ofstream file;
public:
	const int windowWidth = 1536;
	const int windowHeight = 1024;
	void init();
	void display();
	void update(double dt);
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

	float mip_loss_sum;
	float nomip_loss_sum;
	int step;
	ofstream file;
public:
	const int windowWidth = 1536;
	const int windowHeight = 1024;
	void init();
	void display();
	void update(double dt);
};

//original preset

class PresetFilter {
	Matrix mat;
	Matrix hr_mat;

	Attribute target_pos;
	Attribute target_color;

	ProjectParams target_proj;
	RasterizeParams target_rast;
	InterpolateParams target_intr;
	AntialiasParams target_aa;
	FilterParams target_flt1;
	FilterParams target_flt2;

	ProjectParams hr_target_proj;
	RasterizeParams hr_target_rast;
	InterpolateParams hr_target_intr;
	AntialiasParams hr_target_aa;
	GLbuffer gl_hr_target;

	AttributeGrad predict_pos;
	AttributeGrad predict_color;

	ProjectGradParams predict_proj;
	RasterizeGradParams predict_rast;
	InterpolateGradParams predict_intr;
	AntialiasGradParams predict_aa;
	GLbuffer gl_predict;

	ProjectParams hr_predict_proj;
	RasterizeParams hr_predict_rast;
	InterpolateParams hr_predict_intr;
	AntialiasParams hr_predict_aa;
	GLbuffer gl_hr_predict;

	LossParams loss;
	AdamParams adam_pos;
	AdamParams adam_color;

	class Pass {
		AttributeGrad pos;
		AttributeGrad color;

		ProjectGradParams proj;
		RasterizeGradParams rast;
		InterpolateGradParams intr;
		AntialiasGradParams aa;
		GLbuffer gl_aa;
		FilterGradParams flt;
		GLbuffer gl;

		ProjectParams hr_proj;
		RasterizeParams hr_rast;
		InterpolateParams hr_intr;
		AntialiasParams hr_aa;
		GLbuffer gl_hr;

		LossParams loss;
		AdamParams adam_pos;
		AdamParams adam_color;
	public:
		void init(Attribute& predict_pos, Attribute& predict_color, Matrix& mat, FilterParams& target_flt, Matrix& hr_mat, int resolution, int k);
		void forward();
		void draw(float minX, float maxX);
		float getLoss() { return Loss::loss(loss); };
	};

	Pass predict1;
	Pass predict2;
	float loss_sum;
	float loss_sum1;
	float loss_sum2;

	int step;
	ofstream file;

public:
	const int windowWidth = 2048;
	const int windowHeight = 1024;
	void init();
	void display();
	void update(double dt);
};

class PresetPhong {

	Matrix mat;

	Attribute pos;
	Attribute texel;
	Attribute normal;
	Attribute m_pos;
	Attribute r_normal;
	Texture target_texture;
	//Texture white;
	TextureGrad predict_texture;

	NormalcalcParams norm;
	ProjectParams proj;
	RasterizeParams rast;
	InterpolateParams intr;
	ProjectParams pos_proj;
	ProjectParams normal_proj;
	TexturemapParams target_tex;
	MaterialParams target_mtr;
	TexturemapGradParams predict_tex;
	MaterialGradParams predict_mtr;
	//TexturemapParams white_tex;
	//MaterialParams white_mtr;

	LossParams loss;
	AdamParams mtr_adam;
	AdamParams tex_adam;

	GLbuffer predict_buffer;
	GLbuffer target_buffer;
	GLbuffer white_buffer;
	GLbuffer tex_buffer;

	float loss_sum;
	int step;
	ofstream file;

public:
	const int windowWidth = 1024;
	const int windowHeight = 512;
	void init();
	void display(void);
	void update(double dt);
};

class PresetPBR {
	int batch;
	Matrix mat;

	Attribute pos;
	Attribute texel;
	Attribute normal;
	Attribute m_pos;
	Attribute r_normal;
	Texture target_diff;
	Texture target_rough;
	Texture target_nor;
	Texture target_disp;
	TextureGrad predict_diff;
	TextureGrad predict_rough;
	TextureGrad predict_nor;
	TextureGrad predict_disp;

	NormalcalcParams norm;
	ProjectParams proj;
	RasterizeParams rast;
	InterpolateParams intr;
	ProjectParams pos_proj;
	ProjectParams normal_proj;
	TexturemapParams target_diff_tex;
	TexturemapParams target_rough_tex;
	TexturemapParams target_nor_tex;
	TexturemapParams target_disp_tex;
	MaterialParams target_mtr;

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
	float getLoss() { return Loss::loss(loss); };
};