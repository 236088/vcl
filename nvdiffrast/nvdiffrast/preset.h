#pragma once
#include "common.h"
#include "buffer.h"
#include "extention.h"
#include "matrix.h"
#include "project.h"
#include "rasterize.h"
#include "interpolate.h"
#include "texturemap.h"
#include "material.h"
#include "antialias.h"
#include "filter.h"
#include "ComputeNormal.h"
#include "loss.h"
#include "optimize.h"



class PresetPrimitives {
	RotationParams rot;
	CameraParams cam;

	Attribute _pos;
	Attribute _texel;
	Attribute _normal;
	Attribute color;
	Texture target_diffusemap;
	TextureGrad predict_diffusemap;
	Texture _normalmap;
	Texture _roughnessmap;
	SGBuffer target_sgbuf;
	Texture target_sgbake;
	GLbuffer target_bake_buffer;
	SGBufferGrad predict_sgbuf;
	Texture predict_sgbake;
	GLbuffer predict_bake_buffer;

	ProjectParams proj;
	RasterizeParams rast;
	InterpolateParams intr;
	InterpolateParams color_intr;
	InterpolateParams normal_intr;
	TexturemapParams target_diff;
	TexturemapGradParams predict_diff;
	TexturemapParams normalmap;
	TexturemapParams roughnessmap;
	NormalAxisParams normal_axis;
	ViewAxisParams view_axis;
	SGSpecularParams spec;
	SphericalGaussianParams target_sg;
	AntialiasParams target_aa;
	SphericalGaussianGradParams predict_sg;
	AntialiasGradParams predict_aa;

	LossParams loss;
	AdamParams adam_amplitude;
	AdamParams adam_sharpness;
	AdamParams adam_axis;
	AdamParams adam_diffusemap;

	GLbuffer target_map_buffer;
	GLbuffer predict_map_buffer;
	GLbuffer target_aa_buffer;
	GLbuffer predict_aa_buffer;

	Mat out;
	float* buf;
	VideoWriter writer;
	Mat frame;
	Mat frm;
	GLbuffer sample_buffer;

public:
	const int windowWidth = 2048;
	const int windowHeight = 1024;
	void init();
	void display();
	void update(double dt, double t, unsigned int step, bool& play);
};

class PresetPose {
	RotationParams target_rot;
	CameraParams target_cam;

	Attribute target_pos;
	Attribute target_color;
	ProjectParams target_proj;
	RasterizeParams target_rast;
	InterpolateParams target_intr;
	AntialiasParams target_aa;
	GLbuffer gl_target;

	RotationGradParams rot;
	CameraGradParams cam;

	AttributeGrad pos;
	AttributeGrad color;
	ProjectGradParams proj;
	RasterizeGradParams rast;
	InterpolateGradParams intr;
	AntialiasGradParams aa;
	GLbuffer gl;

	LossParams loss;
	float loss_sum;
	double time;

public:
	const int windowWidth = 1024;
	const int windowHeight = 512;
	void init();
	void display();
	void update(double dt, double t, unsigned int step, bool& play);
};
