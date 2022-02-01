#include "preset.h"

#define CONSOLE_INTERVAL 10

void PresetFilter::Pass::init(Attribute& predict_pos,Attribute& predict_color,Matrix& mat, FilterParams& target_flt,Matrix& hr_mat,int resolution, float sigma) {
	AttributeGrad::init(pos, predict_pos.vboNum, predict_pos.vaoNum, 3);
	Attribute::copy(pos, predict_pos);
	AttributeGrad::init(color, pos, 3);
	Attribute::copy(color, predict_color);

	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, resolution, resolution, 1, false);
	Interpolate::init(intr, rast, color);
	Antialias::init(aa, rast, proj, intr.kernel.out, intr.grad.out, 3);
	Filter::init(flt, rast, aa.kernel.out, aa.grad.out, 3, sigma);

	Loss::init(loss, target_flt.kernel.out, flt.kernel.out, flt.grad.out, resolution, resolution, 3);
	Optimizer::init(adam_pos, pos);
	Optimizer::init(adam_color, color);
	Optimizer::init(adam_sigma, flt.kernel.sigma, flt.grad.sigma, 1, 1, 1, 1);
	Adam::setHyperParams(adam_pos, 1e-2, 0.9, 0.999, 1e-8);
	Adam::setHyperParams(adam_color, 1e-2, 0.9, 0.999, 1e-8);
	Adam::setHyperParams(adam_sigma, 1e-1, 0.9, 0.999, 1e-8);

	Project::init(hr_proj, hr_mat.mvp, pos, true);
	Rasterize::init(hr_rast, hr_proj, 512, 512, 1, false);
	Interpolate::init(hr_intr, hr_rast, color);
	Antialias::init(hr_aa, hr_rast, hr_proj, hr_intr.kernel.out, 3);
	Rasterize::wireframeinit(wireframe, hr_proj, 512, 512);

	GLbuffer::init(gl, flt.kernel.out, resolution, resolution, 3);
	GLbuffer::init(gl_hr, hr_aa.kernel.out, 512, 512, 3);
	GLbuffer::init(gl_wireframe, wireframe.kernel.out, 512, 512, 4);
}

void PresetFilter::Pass::forward() {
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Antialias::forward(aa);
	Filter::forward(flt);

	MSELoss::backward(loss);
	Filter::backward(flt);
	Antialias::backward(aa);
	Interpolate::backward(intr);
	Rasterize::backward(rast);
	Project::backward(proj);
	Adam::step(adam_pos);
	Adam::step(adam_color);
	Adam::step(adam_sigma);
	AttributeGrad::clear(pos);
	AttributeGrad::clear(color);

	Project::forward(hr_proj);
	Rasterize::forward(hr_rast);
	Interpolate::forward(hr_intr);
	Antialias::forward(hr_aa);

	Rasterize::drawforward(wireframe);
}

void PresetFilter::Pass::draw(float minX, float maxX) {
	//GLbuffer::draw(gl, GL_RGB32F, GL_RGB, minX, 0.f, maxX, 1.f);
	//GLbuffer::draw(gl_hr, GL_RGB32F, GL_RGB, minX, -1.f, maxX, 0.f);
	GLbuffer::draw(gl, GL_RGB32F, GL_RGB, -.5f, 0.f, 0.f, 1.f);
	GLbuffer::draw(gl_hr, GL_RGB32F, GL_RGB, -.5f, -1.f, 0.f, 0.f);
	GLbuffer::draw(gl_wireframe, GL_RGBA32F, GL_RGBA, 0.f, -1.f, .5f, 0.f);
}

void PresetFilter::init() {
	loss_sum = 0.f;
	step = 0;
	int resolution = 32;
	float sigma = 16.f;
	//file.open("../../filter_log.txt");
	file.open("F:/vcl/picture/filter/filter_log_" + std::to_string(sigma) + ".txt");
	file << "step, nofilter, filter" << std::endl;
	Matrix::init(mat);
	Matrix::setEye(mat, 3.f, 2.f, 3.f);
	Matrix::setFovy(mat, 45.f);
	Attribute::loadOBJ("../../cube.obj", &target_pos, nullptr, nullptr);
	Attribute::init(target_color, target_pos, 3);
	Attribute::copy(target_color, target_pos);
	Attribute::liner(target_color, .5f, .5f);
	AttributeGrad::init(predict_pos, target_pos.vboNum, target_pos.vaoNum, 3);
	Attribute::copy(predict_pos, target_pos);
	Attribute::addRandom(predict_pos, -.5f, .5f);
	AttributeGrad::init(predict_color, predict_pos, 3);
	Attribute::addRandom(predict_color, 0.f, 1.f);

	Project::init(target_proj, mat.mvp, target_pos, true);
	Rasterize::init(target_rast, target_proj, resolution, resolution, 1, false);
	Interpolate::init(target_intr, target_rast, target_color);
	Antialias::init(target_aa, target_rast, target_proj, target_intr.kernel.out, 3);

	Project::init(hr_target_proj, mat.mvp, target_pos, true);
	Rasterize::init(hr_target_rast, hr_target_proj, 512, 512, 1, false);
	Interpolate::init(hr_target_intr, hr_target_rast, target_color);
	Antialias::init(hr_target_aa, hr_target_rast, hr_target_proj, hr_target_intr.kernel.out, 3);

	Project::init(predict_proj, mat.mvp, predict_pos, true);
	Rasterize::init(predict_rast, predict_proj, resolution, resolution, 1, false);
	Interpolate::init(predict_intr, predict_rast, predict_color);
	Antialias::init(predict_aa, predict_rast, predict_proj, predict_intr.kernel.out, predict_intr.grad.out, 3);

	Loss::init(loss, target_aa.kernel.out, predict_aa.kernel.out, predict_aa.grad.out, resolution, resolution, 3);
	Optimizer::init(adam_pos, predict_pos);
	Optimizer::init(adam_color, predict_color);
	Adam::setHyperParams(adam_pos, 1e-2, 0.9, 0.999, 1e-8);
	Adam::setHyperParams(adam_color, 1e-2, 0.9, 0.999, 1e-8);

	Project::init(hr_predict_proj, mat.mvp, predict_pos, true);
	Rasterize::init(hr_predict_rast, hr_predict_proj, 512, 512, 1, false);
	Interpolate::init(hr_predict_intr, hr_predict_rast, predict_color);
	Antialias::init(hr_predict_aa, hr_predict_rast, hr_predict_proj, hr_predict_intr.kernel.out, 3);

	Filter::init(target_flt, target_rast, target_aa.kernel.out, 3, sigma);
	predict1.init(predict_pos, predict_color, mat, target_flt, mat, resolution, 1);

	GLbuffer::init(gl_hr_target, hr_target_aa.kernel.out, 512, 512, 3);
	GLbuffer::init(gl_flt_target, target_flt.kernel.out, resolution, resolution, 3);
	GLbuffer::init(gl_predict, predict_aa.kernel.out, resolution, resolution, 3);
	GLbuffer::init(gl_hr_predict, hr_predict_aa.kernel.out, 512, 512, 3);
}

void PresetFilter::display() {
	Matrix::forward(mat);

	Project::forward(target_proj);
	Rasterize::forward(target_rast);
	Interpolate::forward(target_intr);
	Antialias::forward(target_aa);

	Project::forward(hr_target_proj);
	Rasterize::forward(hr_target_rast);
	Interpolate::forward(hr_target_intr);
	Antialias::forward(hr_target_aa);

	Project::forward(predict_proj);
	Rasterize::forward(predict_rast);
	Interpolate::forward(predict_intr);
	Antialias::forward(predict_aa);

	MSELoss::backward(loss);
	Antialias::backward(predict_aa);
	Interpolate::backward(predict_intr);
	Rasterize::backward(predict_rast);
	Project::backward(predict_proj);
	Adam::step(adam_pos);
	Adam::step(adam_color);
	AttributeGrad::clear(predict_pos);
	AttributeGrad::clear(predict_color);

	Project::forward(hr_predict_proj);
	Rasterize::forward(hr_predict_rast);
	Interpolate::forward(hr_predict_intr);
	Antialias::forward(hr_predict_aa);

	Filter::forward(target_flt);
	predict1.forward();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(gl_predict, GL_RGB32F, GL_RGB, -1.f, 0.f, -.5f, 1.f);
	GLbuffer::draw(gl_hr_predict, GL_RGB32F, GL_RGB, -1.f, -1.f, -.5f, 0.f);
	predict1.draw(0.f, .5f);
	GLbuffer::draw(gl_hr_target, GL_RGB32F, GL_RGB, .5f, -1.f, 1.f, 0.f);
	GLbuffer::draw(gl_flt_target, GL_RGB32F, GL_RGB, .5f, 0.f, 1.f, 1.f);
	glFlush();
}

void PresetFilter::update(double dt, double t, bool& play) {
	loss_sum += Loss::loss(loss);
	loss_sum1 += predict1.getLoss();
	if ((++step) % CONSOLE_INTERVAL == 0) {
		std::cout << step << "," << loss_sum / CONSOLE_INTERVAL <<  "," << loss_sum1 / CONSOLE_INTERVAL <<  std::endl;
		file << step << "," << loss_sum / CONSOLE_INTERVAL << "," << loss_sum1 / CONSOLE_INTERVAL << std::endl;
		loss_sum = 0.f;
		loss_sum1 = 0.f;
	}
	if (step == pause[it]) {
		play = false;
		it++;
	}

	Matrix::setRandomRotation(mat);
}