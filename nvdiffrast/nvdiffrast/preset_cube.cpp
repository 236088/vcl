#include "preset.h"

#define CONSOLE_INTERVAL 10

void PresetCube::init() {
	int resolution = 256;
	predict_loss_sum = 0.f;
	noaa_loss_sum = 0.f;
	step = 0;
	file.open("../../cube_log_"+std::to_string(resolution)+".txt");
	file.open("F:/vcl/picture/cube/cube_log_"+std::to_string(resolution)+".txt");
	file << "step, predict, noAA" << std::endl;

	Matrix::init(mat);
	Matrix::setEye(mat, 0.f, 0.f, 3.5f);
	Matrix::setFovy(mat, 45.f);
	Attribute::loadOBJ("../../cube.obj", &target_pos, nullptr, nullptr);

	Attribute::init(target_color, target_pos, 3);
	Attribute::copy(target_color, target_pos);
	Attribute::liner(target_color, 1.f, .5f);

	Project::init(target_proj, mat.mvp, target_pos, true);
	Rasterize::init(target_rast, target_proj, resolution, resolution, 1, false);
	Interpolate::init(target_intr,target_rast, target_color);
	Antialias::init(target_aa,target_rast, target_proj, target_intr.kernel.out, 3);

	//Project::init(hr_target_proj, mat.mvp, target_pos, true);
	//Rasterize::init(hr_target_rast, hr_target_proj, 512, 512, 1, false);
	//Interpolate::init(hr_target_intr, hr_target_rast, target_color);
	//Antialias::init(hr_target_aa, hr_target_rast, hr_target_proj, hr_target_intr.kernel.out, 3);

	AttributeGrad::init(predict_pos, target_pos.vboNum, target_pos.vaoNum, 3);
	Attribute::copy(predict_pos, target_pos);
	Attribute::addRandom(predict_pos, -.5f, .5f);
	AttributeGrad::init(predict_color, predict_pos, 3);
	Attribute::addRandom(predict_color, 0.f, 1.f);

	Project::init(predict_proj, mat.mvp, predict_pos, true);
	Rasterize::init(predict_rast, predict_proj, resolution, resolution, 1, false);
	Interpolate::init(predict_intr, predict_rast, predict_color);
	Antialias::init(predict_aa, predict_rast, predict_proj, predict_intr.kernel.out, predict_intr.grad.out, 3);

	Loss::init(predict_loss, target_aa.kernel.out, predict_aa.kernel.out, predict_aa.grad.out, resolution, resolution, 3);
	Optimizer::init(predict_adam_pos, predict_pos);
	Optimizer::init(predict_adam_color, predict_color);
	Adam::setHyperParams(predict_adam_pos, 1e-2, 0.9, 0.999, 1e-8);
	Adam::setHyperParams(predict_adam_color, 1e-2, 0.9, 0.999, 1e-8);

	//Project::init(hr_predict_proj, mat.mvp, predict_pos, true);
	//Rasterize::wireframeinit(hr_predict_rast, hr_predict_proj, 512, 512);

	AttributeGrad::init(noaa_pos, predict_pos, 3);
	Attribute::copy(noaa_pos, predict_pos);
	AttributeGrad::init(noaa_color, predict_color, 3);
	Attribute::copy(noaa_color, predict_color);

	Project::init(noaa_proj, mat.mvp, noaa_pos, true);
	Rasterize::init(noaa_rast, noaa_proj, resolution, resolution, 1, false);
	Interpolate::init(noaa_intr, noaa_rast, noaa_color);

	Loss::init(noaa_loss, target_intr.kernel.out, noaa_intr.kernel.out, noaa_intr.grad.out, resolution, resolution, 3);
	Optimizer::init(noaa_adam_pos, noaa_pos);
	Optimizer::init(noaa_adam_color, noaa_color);
	Adam::setHyperParams(noaa_adam_pos, 1e-3, 0.9, 0.999, 1e-8);
	Adam::setHyperParams(noaa_adam_color, 1e-3, 0.9, 0.999, 1e-8);

	Project::init(hr_noaa_proj, mat.mvp, noaa_pos, true);
	Rasterize::wireframeinit(hr_noaa_rast, hr_noaa_proj, 512, 512);

	GLbuffer::init(gl_target, target_aa.kernel.out, resolution, resolution, 3);
	//GLbuffer::init(gl_hr_target, hr_target_aa.kernel.out, 512, 512, 3);
	GLbuffer::init(gl_predict, predict_aa.kernel.out, resolution, resolution, 3);
	//GLbuffer::init(gl_hr_predict, hr_predict_rast.kernel.out, 512, 512, 4);
	GLbuffer::init(gl_noaa, noaa_intr.kernel.out, resolution, resolution, 3);
	//GLbuffer::init(gl_hr_noaa, hr_noaa_rast.kernel.out, 512, 512, 4);
}

void PresetCube::display() {
	Matrix::forward(mat);

	Project::forward(target_proj);
	Rasterize::forward(target_rast);
	Interpolate::forward(target_intr);
	Antialias::forward(target_aa);

	//Project::forward(hr_target_proj);
	//Rasterize::forward(hr_target_rast);
	//Interpolate::forward(hr_target_intr);
	//Antialias::forward(hr_target_aa);

	Project::forward(predict_proj);
	Rasterize::forward(predict_rast);
	Interpolate::forward(predict_intr);
	Antialias::forward(predict_aa);

	MSELoss::backward(predict_loss);
	Antialias::backward(predict_aa);
	Interpolate::backward(predict_intr);
	Rasterize::backward(predict_rast);
	Project::backward(predict_proj);
	Adam::step(predict_adam_pos);
	Adam::step(predict_adam_color);
	AttributeGrad::clear(predict_pos);
	AttributeGrad::clear(predict_color);

	//Project::forward(hr_predict_proj);
	//Rasterize::drawforward(hr_predict_rast);

	Project::forward(noaa_proj);
	Rasterize::forward(noaa_rast);
	Interpolate::forward(noaa_intr);

	MSELoss::backward(noaa_loss);
	Interpolate::backward(noaa_intr);
	Rasterize::backward(noaa_rast);
	Project::backward(noaa_proj);
	Adam::step(noaa_adam_pos);
	Adam::step(noaa_adam_color);
	AttributeGrad::clear(noaa_pos);
	AttributeGrad::clear(noaa_color);

	//Project::forward(hr_noaa_proj);
	//Rasterize::drawforward(hr_noaa_rast);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(gl_target, GL_RGB32F, GL_RGB, -1.f, 0.f, -.33333333f, 1.f);
	//GLbuffer::draw(gl_hr_target, GL_RGB32F, GL_RGB, -1.f, -1.f, -.33333333f, 0.f);
	GLbuffer::draw(gl_predict, GL_RGB32F, GL_RGB, -.33333333f, 0.f, .33333333f, 1.f);
	//GLbuffer::draw(gl_hr_predict, GL_RGBA32F, GL_RGBA, -.33333333f, -1.f, .33333333f, 0.f);
	GLbuffer::draw(gl_noaa, GL_RGB32F, GL_RGB, .33333333f, 0.f, 1.f, 1.f);
	//GLbuffer::draw(gl_hr_noaa, GL_RGBA32F, GL_RGBA, .33333333f, -1.f, 1.f, 0.f);
	glFlush();
}

void PresetCube::update(double dt, double t, bool& play) {
	noaa_loss_sum += Loss::loss(noaa_loss);
	predict_loss_sum += Loss::loss(predict_loss);
	if ((++step) % CONSOLE_INTERVAL == 0) {
		std::cout << step << "," << predict_loss_sum / CONSOLE_INTERVAL << "," << noaa_loss_sum / CONSOLE_INTERVAL << "time:" << t << std::endl;
		file << step << "," << predict_loss_sum / CONSOLE_INTERVAL << "," << noaa_loss_sum / CONSOLE_INTERVAL << std::endl;
		noaa_loss_sum = 0.f;
		predict_loss_sum = 0.f;
	}
	//if (step == pause[it]) {
	if (step == 1000) {
		play = false;
		it++;
	}

	Matrix::setRandomRotation(mat);
}