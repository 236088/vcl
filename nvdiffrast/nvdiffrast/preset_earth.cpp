#include "preset.h"

void PresetEarth::init() {
	batch = 0;
	Matrix::init(mat);
	Matrix::setEye(mat, 0.f, 0.f, 1.5f);
	Matrix::setFovy(mat, 45.f);
	Attribute::loadOBJ("../../sphere.obj", &pos, &texel, nullptr);
	Texture::loadBMP("../../earth-texture.bmp", target_texture, 4);
	TextureGrad::init(predict_texture, target_texture.width, target_texture.height, target_texture.channel, 4);
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, 512, 512, 1, true);
	Interpolate::init(intr, rast, texel);
	Texturemap::init(target_tex, rast, intr, target_texture);
	Antialias::init(target_aa, rast, proj, target_tex.kernel.out, 3);
	Texturemap::init(predict_tex, rast, intr, predict_texture);
	Antialias::init(predict_aa, rast, proj, predict_tex.kernel.out, predict_tex.grad.out, 3);
	Loss::init(loss, target_aa.kernel.out, predict_aa.kernel.out, predict_aa.grad.out, 512, 512, 3);
	Optimizer::init(adam, predict_texture);
	Adam::setHyperParams(adam, 1e-2, 0.9, 0.999, 1e-8);

	GLbuffer::init(gl_target, target_aa.kernel.out, 512, 512, 3, 15);
	GLbuffer::init(gl_predict, predict_aa.kernel.out, 512, 512, 3, 14);
	GLbuffer::init(gl_tex_target, &target_texture.texture[0][2048 * 512 * 3], 2048, 512, 3, 13);
	GLbuffer::init(gl_tex_predict, &predict_texture.texture[0][2048 * 512 * 3], 2048, 512, 3, 12);
}

void PresetEarth::display() {
	batch++;

	Matrix::forward(mat);

	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Texturemap::forward(target_tex);
	Antialias::forward(target_aa);

	Texturemap::forward(predict_tex);
	Antialias::forward(predict_aa);

	MSELoss::backward(loss);
	Antialias::backward(predict_aa);
	Texturemap::backward(predict_tex);
	if (batch % 16 == 0) {
		TextureGrad::gradSumup(predict_texture);
		Adam::step(adam);
		Texture::buildMIP(predict_texture);
		TextureGrad::clear(predict_texture);
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(gl_target, GL_RGB32F, GL_RGB, -1.f, 0.f, -0.6f, 1.f);
	GLbuffer::draw(gl_predict, GL_RGB32F, GL_RGB, -1.f, -1.f, -0.6f, 0.f);
	GLbuffer::draw(gl_tex_target, GL_RGB32F, GL_RGB, -0.6f, 0.f, 1.f, 1.f);
	GLbuffer::draw(gl_tex_predict, GL_RGB32F, GL_RGB, -0.6f, -1.f, 1.f, 0.f);
	glFlush();
}

void PresetEarth::update() {
	Matrix::setRandomRotation(mat);
}