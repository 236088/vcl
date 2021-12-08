#include "preset.h"

void PresetPrimitives::init() {
	int width = 512;
	int height = 512;
	Attribute::loadOBJ("../../simple_sphere.obj", &pos, &texel, nullptr);
	Texture::loadBMP("../../rocks_ground_diff.bmp", diff, 2);
	Texture::loadBMP("../../rocks_ground_rough.bmp", rough, 2);
	Texture::loadBMP("../../rocks_ground_nor.bmp", nor, 2);
	Texture::loadBMP("../../rocks_ground_height.bmp", disp, 2);
	Matrix::init(mat);
	Matrix::setFovy(mat, 30.f);
	Matrix::setEye(mat, 0.f, 1.f, 5.f);
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, width, height, 1, true);
	Interpolate::init(intr, rast, texel);
	Project::init(pos_proj, mat.m, pos, m_pos, false);
	Normalcalc::init(norm, pos, normal);
	Project::init(normal_proj, mat.r, normal, r_normal, false);
	Texturemap::init(diff_tex, rast, intr, diff);
	Texturemap::init(rough_tex, rast, intr, rough);
	Texturemap::init(nor_tex, rast, intr, nor);
	Texturemap::init(disp_tex, rast, intr, disp);
	Material::init(mtr, rast, pos_proj, normal_proj, &texel, 3, diff_tex.kernel.out, rough_tex.kernel.out, nor_tex.kernel.out, disp_tex.kernel.out);
	float3 direction[4]{
		2.f,3.f,1.f,
		-3.f,2.f,2.f,
		-2.f,-3.f,1.f,
		3.f,-2.f,2.f,
	};
	float3 lightintensity[4]{
		1.f,1.f,1.f,
		2.f,2.f,2.f,
		2.f,2.f,2.f,
		1.f,1.f,1.f,
	};
	PBRMaterial::init(mtr, (float3*)&mat.eye, 4, direction, lightintensity, 1.53f);
	Antialias::init(aa, rast, proj,  mtr.kernel.out, 3);
	GaussianFilter::init(flt, rast, aa.kernel.out, 3, 16);

	GLbuffer::init(rast_buffer, rast.kernel.out, width, height, 4, 15);
	GLbuffer::init(intr_buffer, intr.kernel.out, width, height, 2, 14);
	GLbuffer::init(diff_buffer, diff_tex.kernel.out, width, height, 3, 13);
	GLbuffer::init(mtr_buffer, mtr.kernel.out, width, height, 3, 12);
	GLbuffer::init(aa_buffer, aa.kernel.out, width, height, 3, 11);
	GLbuffer::init(flt_buffer, flt.kernel.out, width, height, 3, 10);
}

void PresetPrimitives::display(void) {
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Project::forward(pos_proj);
	Normalcalc::forward(norm);
	Project::forward(normal_proj);
	Texturemap::forward(diff_tex);
	Texturemap::forward(rough_tex);
	Texturemap::forward(nor_tex);
	Texturemap::forward(disp_tex);
	PBRMaterial::forward(mtr);
	Antialias::forward(aa);
	Filter::forward(flt);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(rast_buffer, GL_RG32F, GL_RGBA, -1.f, 0.f, -.33333333f, 1.f);
	GLbuffer::draw(intr_buffer, GL_RG32F, GL_RG, -.33333333f, 0.f, .33333333f, 1.f);
	GLbuffer::draw(diff_buffer, GL_RGB32F, GL_RGB, .33333333f, 0.f, 1.f, 1.f);
	GLbuffer::draw(mtr_buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, -.33333333f, 0.f);
	GLbuffer::draw(aa_buffer, GL_RGB32F, GL_RGB, -.33333333f, -1.f, .33333333f, 0.f);
	GLbuffer::draw(flt_buffer, GL_RGB32F, GL_RGB, .33333333f, -1.f, 1.f, 0.f);
	glFlush();
}

void PresetPrimitives::update(void) {
	Matrix::addRotation(mat, 1.f, 0.f, 1.f, 0.f);
}