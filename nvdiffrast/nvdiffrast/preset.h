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
#include "normalcalc.h"
#include "loss.h"
#include "optimize.h"



class PresetPrimitives {
	RotationParams rot;
	CameraParams cam;

	Attribute _pos;
	Attribute _texel;
	Attribute _normal;
	Attribute color;
	Texture _diffusemap;
	Texture _normalmap;
	Texture _roughnessmap;
	SGBuffer sgbuf;
	Texture sgbake;
	GLbuffer bake_buffer;

	ProjectParams proj;
	RasterizeParams rast;
	InterpolateParams intr;
	InterpolateParams color_intr;
	InterpolateParams normal_intr;
	TexturemapParams diffusemap;
	TexturemapParams normalmap;
	TexturemapParams roughnessmap;
	NormalAxisParams normal_axis;
	ReflectAxisParams reflect_axis;
	SphericalGaussianParams sg;
	AntialiasParams aa;
	FilterParams flt;

	RasterizeParams wireframe;
	RasterizeParams idhash;

	GLbuffer rast_buffer;
	GLbuffer intr_buffer;
	GLbuffer color_buffer;
	GLbuffer tex_buffer;
	GLbuffer aa_buffer;
	GLbuffer flt_buffer;
	GLbuffer normal_buffer;
	GLbuffer sgdiffenv_buffer;
	GLbuffer sgspecenv_buffer;
	GLbuffer sg_buffer;

	GLbuffer sample_buffer;


public:
	const int windowWidth = 2048;
	const int windowHeight = 1024;
	void init();
	void display();
	void update(double dt, double t, unsigned int step, bool& play);
};