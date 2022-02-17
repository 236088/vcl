#include "preset.h"

void PresetPrimitives::init() {
	int width = 1024;
	int height = 1024;
	Attribute::loadOBJ("../../monkey.obj", &pos, &texel, &normal);
	Attribute::init(color, pos, 3);
	Attribute::addRandom(color, 0.f, 1.f);
	Attribute::step(color, .5f);
	Texture::loadBMP("../../uvtemplate.bmp", mip_texture, 8);
	//Texture::loadBMP("../../uvtemplate.bmp", texture, 1);
	Matrix::init(mat);
	Matrix::setRotation(mat, 30.f, 0.f, 1.f, 0.f);
	Matrix::setFovy(mat, 30.f);
	Matrix::setEye(mat, 0.f, 0.f, 5.f);
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, width, height, 1, true);
	Interpolate::init(intr, rast, texel);
	//Interpolate::init(color_intr, rast, color);
	Texturemap::init(mip_tex, rast, intr, mip_texture);
	//Texturemap::init(tex, rast, intr, texture);
	//Project::init(pos_proj, mat.m, pos, false);
	//Project::init(normal_proj, mat.r, normal, false);
	float _point[12]{
		2.f,2.f,2.f,
		2.f,-2.f,-2.f,
		-2.f,2.f,-2.f,
		-2.f,-2.f,2.f,
	};
	float _intensity[12]{
		20.f,20.f,20.f,
		1.f,1.f,1.f,
		1.f,1.f,1.f,
		1.f,1.f,1.f,
	};
	float _params[4]{
		.3f, .5f, .7f, 10.f
	};
	//Buffer::init(point, 1, 3);
	//Buffer::init(intensity, 1, 3);
	//Buffer::init(params, 4, 1);
	//Buffer::copy(point, _point);
	//Buffer::copy(intensity, _intensity);
	//Buffer::copy(params, _params);
	//Material::init(mtr, rast, pos_proj, normal_proj, &texel, 3, mip_tex.kernel.out);
	//Material::init(mtr, *(float3*)&mat.eye, point, intensity);
	//Material::init(mtr, params);
	Antialias::init(aa, rast, proj, mip_tex.kernel.out, 3);
	Filter::init(flt, rast, aa.kernel.out, 3, 60);
	//Rasterize::wireframeinit(wireframe, proj, width, height);
	//Rasterize::idhashinit(idhash, proj, width, height);

	//GLbuffer::init(rast_buffer, rast.kernel.out, width, height, 4);
	//GLbuffer::init(intr_buffer, intr.kernel.out, width, height, 2);
	//GLbuffer::init(color_buffer, color_intr.kernel.out, width, height, 3);
	//GLbuffer::init(tex_buffer, tex.kernel.out, width, height, 3);
	//GLbuffer::init(mip_tex_buffer, mip_tex.kernel.out, width, height, 3);
	//GLbuffer::init(mtr_buffer, mtr.kernel.out, width, height, 3);
	//GLbuffer::init(aa_buffer, aa.kernel.out, width, height, 3);
	GLbuffer::init(flt_buffer, flt.kernel.out, width, height, 3);
	//GLbuffer::init(wireframe_buffer, wireframe.kernel.out, width, height, 4);
	//GLbuffer::init(idhash_buffer, idhash.kernel.out, width, height, 4);
}

void PresetPrimitives::display(void) {
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	//Interpolate::forward(color_intr);
	//Texturemap::forward(tex);
	Texturemap::forward(mip_tex);
	//Project::forward(pos_proj);
	//Project::forward(normal_proj);
	//PhongMaterial::forward(mtr);
	Antialias::forward(aa);
	Filter::forward(flt);
	//Rasterize::drawforward(wireframe);
	//Rasterize::drawforward(idhash);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	//GLbuffer::draw(rast_buffer, GL_RG32F, GL_RGBA, -1.f, 0.f, -.5f, 1.f);
	//GLbuffer::draw(intr_buffer, GL_RG32F, GL_RG, -.5f, 0.f, 0.f, 1.f);
	//GLbuffer::draw(tex_buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, -.5f, 0.f);
	//GLbuffer::draw(mtr_buffer, GL_RGB32F, GL_RGB, -.5f, -1.f, 0.f, 0.f);
	//GLbuffer::draw(aa_buffer, GL_RGB32F, GL_RGB, 0.f, -1.f, .5f, 0.f);
	//GLbuffer::draw(flt_buffer, GL_RGB32F, GL_RGB, .5f, -1.f, 1.f, 0.f);
	//GLbuffer::draw(wireframe_buffer, GL_RGB32F, GL_RGBA, 0.f, 0.f, .5f, 1.f);
	GLbuffer::draw(flt_buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, 0.f, 1.f);
	//GLbuffer::draw(mip_tex_buffer, GL_RGB32F, GL_RGB, 0.f, -1.f, 1.f, 1.f);
	glFlush();
}

void PresetPrimitives::update(double dt, double t, bool& play) {
	Matrix::addRotation(mat, .25f, 0.f, 1.f, 0.f);
}