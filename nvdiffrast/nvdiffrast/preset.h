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


class PresetPrimitives {
	Matrix mat;

	Attribute pos;
	Attribute texel;
	Attribute normal;
	Attribute color;
	Texture mip_texture;
	Texture texture;
	Buffer point;
	Buffer intensity;
	Buffer params;

	ProjectParams proj;
	RasterizeParams rast;
	InterpolateParams intr;
	InterpolateParams color_intr;
	ProjectParams pos_proj;
	ProjectParams normal_proj;
	TexturemapParams mip_tex;
	TexturemapParams tex;
	MaterialParams mtr;
	AntialiasParams aa;
	FilterParams flt;
	RasterizeParams wireframe;
	RasterizeParams idhash;

	GLbuffer rast_buffer;
	GLbuffer intr_buffer;
	GLbuffer color_buffer;
	GLbuffer tex_buffer;
	GLbuffer mip_tex_buffer;
	GLbuffer mtr_buffer;
	GLbuffer aa_buffer;
	GLbuffer flt_buffer;
	GLbuffer wireframe_buffer;
	GLbuffer idhash_buffer;

public:
	const int windowWidth = 2048;
	const int windowHeight = 1024;
	void init();
	void display();
	void update(double dt, double t, bool& play);
};

class PresetCube {
	Matrix mat;

	Attribute target_pos;
	Attribute target_color;
	ProjectParams target_proj;
	RasterizeParams target_rast;
	InterpolateParams target_intr;
	AntialiasParams target_aa;
	GLbuffer gl_target;

	AttributeGrad pos;
	AttributeGrad color;
	ProjectGradParams proj;
	RasterizeGradParams rast;
	InterpolateGradParams intr;
	AntialiasGradParams aa;
	GLbuffer gl;

	ProjectParams hr_target_proj;
	RasterizeParams hr_target_rast;
	InterpolateParams hr_target_intr;
	AntialiasParams hr_target_aa;
	GLbuffer gl_hr_target;

	ProjectParams hr_proj;
	RasterizeParams hr_rast;
	InterpolateParams hr_intr;
	AntialiasParams hr_aa;
	GLbuffer gl_hr;

	LossParams loss;
	AdamParams adam_pos;
	AdamParams adam_color;
	float loss_sum;
	float error_sum;
	double time;

	int step;
	ofstream file;
	int pause[10]{ 10,20, 50, 100,200,500,1000,2000,5000 ,10000 };
	int it = 0;

public:
	const int windowWidth = 1024;
	const int windowHeight = 1024;
	void init();
	void display();
	void update(double dt, double t, bool& play);
};

class PresetEarth {
	Matrix mat;
	Attribute pos;
	Attribute texel;
	ProjectParams proj;
	RasterizeParams target_rast;
	InterpolateParams target_intr;
	Texture target_texture;
	TexturemapParams target_tex;
	Texture out_tex;
	GLbuffer gl_tex_target;
	GLbuffer gl_target;

	RasterizeParams rast;
	InterpolateParams intr;

	TextureGrad texture;
	TexturemapGradParams tex;
	LossParams loss;
	AdamParams adam;
	LossParams tex_loss;
	GLbuffer gl_tex;
	GLbuffer gl;

	float loss_sum;
	float error_sum;
	double time;

	int step;
	ofstream file;
	int pause[8]{ 100, 200, 500, 1000, 2000, 5000,10000 ,20000 };
	int it = 0;

public:
	const int windowWidth = 1024;
	const int windowHeight = 1024;
	void init();
	void display();
	void update(double dt, double t, bool& play);
};

//original preset

class PresetFilter {
	Matrix mat;
	float target_sigma;
	float sigma;

	Attribute target_pos;
	Attribute target_color;
	ProjectParams target_proj;
	RasterizeParams target_rast;
	InterpolateParams target_intr;
	AntialiasParams target_aa;
	FilterParams target_flt;
	GLbuffer gl_aa_target;
	GLbuffer gl_target;

	AttributeGrad pos;
	AttributeGrad color;
	ProjectGradParams proj;
	RasterizeGradParams rast;
	InterpolateGradParams intr;
	AntialiasGradParams aa;
	FilterGradParams flt;
	GLbuffer gl_aa;
	GLbuffer gl;

	ProjectParams hr_target_proj;
	RasterizeParams hr_target_rast;
	InterpolateParams hr_target_intr;
	AntialiasParams hr_target_aa;
	GLbuffer gl_hr_target;

	ProjectParams hr_proj;
	RasterizeParams hr_rast;
	InterpolateParams hr_intr;
	AntialiasParams hr_aa;
	GLbuffer gl_hr;

	LossParams loss;
	AdamParams adam_pos;
	AdamParams adam_color;
	AdamParams adam_sigma;
	float loss_sum;
	float error_sum;
	double time;

	int step;
	ofstream file;
	int pause[10]{ 10,20, 50, 100,200,500,1000,2000,5000 ,10000 };
	int it = 0;

public:
	const int windowWidth = 1536;
	const int windowHeight = 1024;
	void init();
	void display();
	void update(double dt, double t, bool& play);
};

class PresetPhong {

	Matrix mat;

	Attribute pos;
	Attribute texel;
	Attribute normal;
	Attribute m_pos;
	Attribute r_normal;
	Texture target_texture;
	Buffer target_point;
	Buffer target_intensity;
	Buffer target_params;
	BufferGrad point;
	BufferGrad intensity;
	BufferGrad params;

	ProjectParams proj;
	RasterizeParams rast;
	InterpolateParams intr;
	ProjectParams pos_proj;
	ProjectParams normal_proj;
	TexturemapParams target_tex;
	MaterialParams target_mtr;
	MaterialGradParams mtr;

	LossParams loss;
	AdamParams point_adam;
	AdamParams intensity_adam;
	AdamParams params_adam;

	GLbuffer buffer;
	GLbuffer target_buffer;

	float loss_sum;
	float* h_params;
	double time;

	int step;
	ofstream file;
	int pause[6]{ 10,100,1000,2000,5000,10000, };
	int it = 0;

public:
	const int windowWidth = 1024;
	const int windowHeight = 512;
	void init();
	void display(void);
	void update(double dt, double t, bool& play);
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
	TextureGrad diff;
	TextureGrad rough;
	TextureGrad nor;
	TextureGrad disp;

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

	TexturemapGradParams diff_tex;
	TexturemapGradParams rough_tex;
	TexturemapGradParams nor_tex;
	TexturemapGradParams disp_tex;
	MaterialGradParams mtr;

	GLbuffer target_diff_buffer;
	GLbuffer target_rough_buffer;
	GLbuffer target_nor_buffer;
	GLbuffer target_mtr_buffer;
	GLbuffer diff_buffer;
	GLbuffer rough_buffer;
	GLbuffer nor_buffer;
	GLbuffer mtr_buffer;

	LossParams loss;
	AdamParams diff_adam;
	AdamParams rough_adam;
	AdamParams nor_adam;
	int pause[6]{ 100,200,500,1000,2000,5000 };
	int it = 0;


public:
	const int windowWidth = 2048;
	const int windowHeight = 1024;
	double t;
	void init();
	void display(void);
	void update(double dt, double t, bool& play);
	float getLoss() { return Loss::loss(loss); };
};