#include "preset.h"

#define CONSOLE_INTERVAL 10


void PresetEarth::init() {
	mip_loss_sum = 0.f;
	nomip_loss_sum = 0.f;
	step = 0;
	//file.open("../../earth_log.txt");
	file.open("F:/vcl/picture/earth/earth_log.txt");
	file << "step, predict, noMIP" << std::endl;
	Attribute::loadOBJ("../../sphere.obj", &pos, &texel, nullptr);
	Matrix::init(mat);
	Matrix::setEye(mat, 0.f, 0.f, 4.f);
	Matrix::setFovy(mat, 30.f);
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(target_rast, proj, 4096, 4096, 1, true);
	Interpolate::init(target_intr, target_rast, texel);
	Texture::loadBMP("../../earth-texture.bmp", target_texture, 1);
	Texturemap::init(target_tex, target_rast, target_intr, target_texture);
	Texture::init(out_tex, target_tex.kernel.out, 4096, 4096, 3, 4);

	Rasterize::init(rast, proj, 512, 512, 1, true);
	Interpolate::init(intr, rast, texel);

	TextureGrad::init(predict_texture, target_texture.width, target_texture.height, target_texture.channel, 1);
	Texturemap::init(predict_tex, rast, intr, predict_texture);
	Loss::init(loss, out_tex.texture[3], predict_tex.kernel.out, predict_tex.grad.out, 512, 512, 3);
	Optimizer::init(adam, predict_texture);
	Adam::setHyperParams(adam, 1e-3, 0.9, 0.99, 1e-8);
	Loss::init(tex_loss, target_texture.texture[0], predict_texture.texture[0], nullptr, target_texture.width, target_texture.height, target_texture.channel);

	TextureGrad::init(predict_mip_texture, target_texture.width, target_texture.height, target_texture.channel, 3);
	Texturemap::init(predict_mip_tex, rast, intr, predict_mip_texture);
	Loss::init(mip_loss, out_tex.texture[3], predict_mip_tex.kernel.out, predict_mip_tex.grad.out, 512, 512, 3);
	Optimizer::init(mip_adam, predict_mip_texture);
	Adam::setHyperParams(mip_adam, 1e-3, 0.9, 0.99, 1e-8);
	Loss::init(mip_tex_loss, target_texture.texture[0], predict_mip_texture.texture[0], nullptr, target_texture.width, target_texture.height, target_texture.channel);

	GLbuffer::init(gl_predict, predict_tex.kernel.out, 512, 512, 3);
	GLbuffer::init(gl_tex_predict, &predict_texture.texture[0][2048 * 512 * 3], 2048, 512, 3);
	GLbuffer::init(gl_mip_predict, predict_mip_tex.kernel.out, 512, 512, 3);
	GLbuffer::init(gl_tex_mip_predict, &predict_mip_texture.texture[0][2048 * 512 * 3], 2048, 512, 3);
	GLbuffer::init(gl_target, out_tex.texture[3], 512, 512, 3);
	GLbuffer::init(gl_tex_target, &target_texture.texture[0][2048 * 512 * 3], 2048, 512, 3);
}

void PresetEarth::display() {
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(target_rast);
	Interpolate::forward(target_intr);
	Texturemap::forward(target_tex);
	Texture::bilinearDownsampling(out_tex);

	Rasterize::forward(rast);
	Interpolate::forward(intr);

	Texturemap::forward(predict_tex);
	MSELoss::backward(loss);
	Texturemap::backward(predict_tex);
	Adam::step(adam);
	MSELoss::textureloss(tex_loss);
	TextureGrad::clear(predict_texture);

	Texturemap::forward(predict_mip_tex);
	MSELoss::backward(mip_loss);
	Texturemap::backward(predict_mip_tex);
	TextureGrad::gradSumup(predict_mip_texture);
	Adam::step(mip_adam);
	Texture::buildMIP(predict_mip_texture);
	MSELoss::textureloss(mip_tex_loss);
	TextureGrad::clear(predict_mip_texture);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(gl_predict, GL_RGB32F, GL_RGB, -1.f, 0.f, -.33333333f, 1.f);
	GLbuffer::draw(gl_tex_predict, GL_RGB32F, GL_RGB, .125f, .5f, .1875f, .75f, -1.f, -1.f, -.33333333f, 0.f);
	GLbuffer::draw(gl_mip_predict, GL_RGB32F, GL_RGB,  -.33333333f, 0.f, .33333333f, 1.f);
	GLbuffer::draw(gl_tex_mip_predict, GL_RGB32F, GL_RGB, .125f, .5f, .1875f, .75f, -.33333333f, -1.f, .33333333f, 0.f);
	GLbuffer::draw(gl_target, GL_RGB32F, GL_RGB, .33333333f, 0.f, 1.f, 1.f);
	GLbuffer::draw(gl_tex_target, GL_RGB32F, GL_RGB, .125f,  .5f, .1875f, .75f, .33333333f, -1.f, 1.f, 0.f);
	glFlush();
}

void PresetEarth::update(double dt, double t, bool& play) {
	mip_loss_sum -=10.0 * log10( Loss::loss(mip_tex_loss));
	nomip_loss_sum -= 10.0 * log10(Loss::loss(tex_loss));
	if ((++step) % CONSOLE_INTERVAL == 0) {
		mip_loss_sum /= CONSOLE_INTERVAL;
		nomip_loss_sum /= CONSOLE_INTERVAL;
		std::cout << step << "," << mip_loss_sum << "," << nomip_loss_sum << std::endl;
		file << step << "," <<  mip_loss_sum << "," << nomip_loss_sum << std::endl;
		nomip_loss_sum = 0.f;
		mip_loss_sum = 0.f;
	}	
	if (step == pause[it]) {
		play = false;
		it++;
	}
	Matrix::setRandomRotation(mat);
	float x = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
	float y = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
	float z = (float)rand() / (float)RAND_MAX * 48.5f + 1.5f;
	Matrix::setEye(mat, 0.f, 0.f, z);
	Matrix::setOrigin(mat, x, y, 0.f);
}