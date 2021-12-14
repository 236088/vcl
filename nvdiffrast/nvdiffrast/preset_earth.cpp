#include "preset.h"

void PresetEarth::Pass::init(RasterizeParams& rast, InterpolateParams& intr, int miplevel) {
	Texture::loadBMP("../../earth-texture.bmp", target_texture, miplevel);
	TextureGrad::init(predict_texture, target_texture.width, target_texture.height, target_texture.channel, miplevel);
	
	Texturemap::init(target_tex, rast, intr, target_texture);
	Texturemap::init(predict_tex, rast, intr, predict_texture);
	Loss::init(loss, target_tex.kernel.out, predict_tex.kernel.out, predict_tex.grad.out, 512, 512, 3);
	Optimizer::init(adam, predict_texture);
	Adam::setHyperParams(adam, 1e-3, 0.9, 0.999, 1e-8);
}

void PresetEarth::Pass::forward() {
	Texturemap::forward(target_tex);
	Texturemap::forward(predict_tex);
	MSELoss::backward(loss);
	Texturemap::backward(predict_tex);
	if (predict_tex.kernel.miplevel > 1)TextureGrad::gradSumup(predict_texture);
	Adam::step(adam);
	if (predict_tex.kernel.miplevel > 1)Texture::buildMIP(predict_texture);
	TextureGrad::clear(predict_texture);
}

void PresetEarth::init() {
	Attribute::loadOBJ("../../sphere.obj", &pos, &texel, nullptr);
	Matrix::init(mat);
	Matrix::setEye(mat, 0.f, 0.f, 4.f);
	Matrix::setFovy(mat, 30.f);
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, 512, 512, 1, true);
	Interpolate::init(intr, rast, texel);
	//nomip.init(rast, intr, 1);
	mip.init(rast, intr, 8);
	GLbuffer::init(gl_target, mip.target_tex.kernel.out, 512, 512, 3, 15);
	GLbuffer::init(gl_tex_target, &mip.target_texture.texture[0][2048 * 512 * 3], 2048, 512, 3, 13);
	//GLbuffer::init(gl_predict, nomip.predict_tex.kernel.out, 512, 512, 3, 12);
	//GLbuffer::init(gl_tex_predict, &nomip.predict_texture.texture[0][2048 * 512 * 3], 2048, 512, 3, 11);
	GLbuffer::init(gl_mip_predict, mip.predict_tex.kernel.out, 512, 512, 3, 10);
	GLbuffer::init(gl_tex_mip_predict, &mip.predict_texture.texture[0][2048 * 512 * 3], 2048, 512, 3, 9);
}

void PresetEarth::display() {
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	mip.forward();
	//nomip.forward();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	//GLbuffer::draw(gl_predict, GL_RGB32F, GL_RGB, -1.f, 0.f, -.33333333f, 1.f);
	//GLbuffer::draw(gl_tex_predict, GL_RGB32F, GL_RGB, .25f, 0.f, .5f, 1.f, -1.f, -1.f, -.33333333f, 0.f);
	//GLbuffer::draw(gl_mip_predict, GL_RGB32F, GL_RGB,  -.33333333f, 0.f, .33333333f, 1.f);
	//GLbuffer::draw(gl_tex_mip_predict, GL_RGB32F, GL_RGB, .25f, 0.f, .5f, 1.f, -.33333333f, -1.f, .33333333f, 0.f);
	//GLbuffer::draw(gl_target, GL_RGB32F, GL_RGB, .33333333f, 0.f, 1.f, 1.f);
	//GLbuffer::draw(gl_tex_target, GL_RGB32F, GL_RGB, .25f, 0.f, .5f, 1.f, .33333333f, -1.f, 1.f, 0.f);
	GLbuffer::draw(gl_mip_predict, GL_RGB32F, GL_RGB,  -1.f, 0.f, -.33333333f, 1.f);
	GLbuffer::draw(gl_tex_mip_predict, GL_RGB32F, GL_RGB, 0.f, 0.f, .5f, 1.f, -.33333333f, 0.f, 1.f, 1.f);
	GLbuffer::draw(gl_target, GL_RGB32F, GL_RGB, -1.f, -1.f, -.33333333f, 0.f);
	GLbuffer::draw(gl_tex_target, GL_RGB32F, GL_RGB, 0.f, 0.f, .5f, 1.f, -.33333333f, -1.f, 1.f, 0.f);
	glFlush();
}

void PresetEarth::update() {
	Matrix::setRandomRotation(mat);
}