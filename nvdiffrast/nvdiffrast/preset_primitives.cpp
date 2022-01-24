#include "preset.h"

void PresetPrimitives::init() {
	int width = 256;
	int height = 256;
	Attribute::loadOBJ("../../monkey.obj", &pos, &texel, &normal);
	Texture::loadBMP("../../uvtemplate.bmp", texture, 8);
	Matrix::init(mat);
	Matrix::setRotation(mat, 30.f, 0.f, 1.f, 0.f);
	Matrix::setFovy(mat, 30.f);
	Matrix::setEye(mat, 0.f, 0.f, 5.f);
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, width, height, 1, true);
	Interpolate::init(intr, rast, texel);
	Texturemap::init(tex, rast, intr, texture);
	Project::init(pos_proj, mat.m, pos, false);
	Project::init(normal_proj, mat.r, normal, false);
	Material::init(mtr, rast, pos_proj, normal_proj, &texel, 3, tex.kernel.out);
	float3 point[8]{
		2.f,2.f,2.f,
		-2.f,2.f,2.f,
		2.f,-2.f,2.f,
		-2.f,-2.f,2.f,
		2.f,2.f,-2.f,
		-2.f,2.f,-2.f,
		2.f,-2.f,-2.f,
		-2.f,-2.f,-2.f,
	};
	float intensity[24]{
		0.f,0.f,0.f,
		1.f,1.f,0.f,
		1.f,0.f,1.f,
		0.f,1.f,1.f,
		1.f,0.f,0.f,
		0.f,1.f,0.f,
		0.f,0.f,1.f,
		1.f,1.f,1.f,
	};
	PhongMaterial::init(mtr, *(float3*)&mat.eye, 8, point, intensity,  .3f, .5f, .7f, 10.f);
	Antialias::init(aa, rast, proj, mtr.kernel.out, 3);
	Filter::init(flt, rast, aa.kernel.out, 3, 4);

	GLbuffer::init(rast_buffer, rast.kernel.out, width, height, 4);
	GLbuffer::init(intr_buffer, intr.kernel.out, width, height, 2);
	GLbuffer::init(tex_buffer, tex.kernel.out, width, height, 3);
	GLbuffer::init(mtr_buffer, mtr.kernel.out, width, height, 3);
	GLbuffer::init(aa_buffer, aa.kernel.out, width, height, 3);
	GLbuffer::init(flt_buffer, flt.kernel.out, width, height, 3);
}

void PresetPrimitives::display(void) {
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Texturemap::forward(tex);
	Project::forward(pos_proj);
	Project::forward(normal_proj);
	PhongMaterial::forward(mtr);
	Antialias::forward(aa);
	Filter::forward(flt);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(rast_buffer, GL_RG32F, GL_RGBA, -1.f, 0.f, -.5f, 1.f);
	GLbuffer::draw(intr_buffer, GL_RG32F, GL_RG, -.5f, 0.f, 0.f, 1.f);
	GLbuffer::draw(tex_buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, -.5f, 0.f);
	GLbuffer::draw(mtr_buffer, GL_RGB32F, GL_RGB, -.5f, -1.f, 0.f, 0.f);
	GLbuffer::draw(aa_buffer, GL_RGB32F, GL_RGB, 0.f, -1.f, .5f, 0.f);
	GLbuffer::draw(flt_buffer, GL_RGB32F, GL_RGB, .5f, -1.f, 1.f, 0.f);
	glFlush();
}

void PresetPrimitives::update(double dt) {
	Matrix::addRotation(mat, .25f, 0.f, 1.f, 0.f);
}