#include "preset.h"

void PresetPrimitives::init() {
	int width = 256;
	int height = 256;
	Attribute::loadOBJ("../../monkey.obj", &pos, &texel, nullptr);
	Texture::loadBMP("../../rocks_ground_diff.bmp", texture, 8);
	Matrix::init(mat);
	Matrix::setFovy(mat, 45);
	Matrix::setEye(mat, 0.f, .5f, 4.f);
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, width, height, 1, true);
	Interpolate::init(intr, rast, texel);
	Project::init(pos_proj, mat.m, pos, m_pos, false);
	Interpolate::init(pos_intr, rast, m_pos);
	Normalcalc::init(norm, pos, normal);
	Project::init(normal_proj, mat.r, normal, r_normal, false);
	Interpolate::init(normal_intr, rast, r_normal);
	Texturemap::init(tex, rast, intr, texture);
	Material::init(mtr, rast, pos_intr, normal_intr, 3, tex.kernel.out);
	float3 direction[4]{
		-1.f, 0.f, -2.f,
		2.f, 0.f, -2.f,
		0.f, -2.f, 1.f,
		0.f, -1.f, 2.f,
	};
	float3 lightintensity[4]{
		.7f,.7f,.7f,
		.5f,.5f,.5f,
		.3f,.3f,.3f,
		.9f,.9f,.9f,
	};
	PBRMaterial::init(mtr, (float3*)&mat.eye, 4, direction, lightintensity, .99f, 2.417f);
	Antialias::init(aa, rast, proj,  mtr.kernel.out, 3);
	GaussianFilter::init(flt, rast, aa.kernel.out, 3, 16);

	GLbuffer::init(rast_buffer, rast.kernel.out, width, height, 4, 15);
	GLbuffer::init(intr_buffer, intr.kernel.out, width, height, 2, 14);
	GLbuffer::init(pos_intr_buffer, pos_intr.kernel.out, width, height, 3, 13);
	GLbuffer::init(normal_intr_buffer, normal_intr.kernel.out, width, height, 3, 12);
	GLbuffer::init(tex_buffer, tex.kernel.out, width, height, 3, 11);
	GLbuffer::init(mtr_buffer, mtr.kernel.out, width, height, 3, 10);
	GLbuffer::init(aa_buffer, aa.kernel.out, width, height, 3, 9);
	GLbuffer::init(flt_buffer, flt.kernel.out, width, height, 3, 8);
}

void PresetPrimitives::display(void) {
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Project::forward(pos_proj);
	Interpolate::forward(pos_intr);
	Normalcalc::forward(norm);
	Project::forward(normal_proj);
	Interpolate::forward(normal_intr);
	Texturemap::forward(tex);
	PBRMaterial::forward(mtr);
	Antialias::forward(aa);
	Filter::forward(flt);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(rast_buffer, GL_RG32F, GL_RGBA, -1.f, 0.f, -.5f, 1.f);
	GLbuffer::draw(intr_buffer, GL_RG32F, GL_RG, -.5f, 0.f, 0.f, 1.f);
	GLbuffer::draw(pos_intr_buffer, GL_RGB32F, GL_RGB, 0.f, 0.f, .5f, 1.f);
	GLbuffer::draw(normal_intr_buffer, GL_RGB32F, GL_RGB, .5f, 0.f, 1.f, 1.f);
	GLbuffer::draw(tex_buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, -.5f, 0.f);
	GLbuffer::draw(mtr_buffer, GL_RGB32F, GL_RGB, -.5f, -1.f, 0.f, 0.f);
	GLbuffer::draw(aa_buffer, GL_RGB32F, GL_RGB, 0.f, -1.f, .5f, 0.f);
	GLbuffer::draw(flt_buffer, GL_RGB32F, GL_RGB, .5f, -1.f, 1.f, 0.f);
	glFlush();
}

void PresetPrimitives::update(void) {
	Matrix::addRotation(mat, .25f, 0.f, 1.f, 0.f);
}