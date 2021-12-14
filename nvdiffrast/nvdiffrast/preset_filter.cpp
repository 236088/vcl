#include "preset.h"

void PresetFilter::init(int resolution,int k) {
	Matrix::init(mat);
	Matrix::setEye(mat, 3.f, 2.f, 3.f);
	Matrix::setFovy(mat, 45.f);
	Matrix::init(hr_mat);
	Matrix::setEye(hr_mat, 3.f, 2.f, 3.f);
	Matrix::setFovy(hr_mat, 45.f);
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
	GaussianFilter::init(target_flt, target_rast, target_aa.kernel.out, 3, k);

	Project::init(predict_proj, mat.mvp, predict_pos, true);
	Rasterize::init(predict_rast, predict_proj, resolution, resolution, 1, false);
	Interpolate::init(predict_intr, predict_rast, predict_color);
	Antialias::init(predict_aa, predict_rast, predict_proj, predict_intr.kernel.out, predict_intr.grad.out, 3);
	GaussianFilter::init(predict_flt, predict_rast, predict_aa.kernel.out, predict_aa.grad.out, 3, k);

	Loss::init(loss, target_flt.kernel.out, predict_flt.kernel.out, predict_flt.grad.out, resolution, resolution, 3);
	Optimizer::init(adam_pos, predict_pos);
	Optimizer::init(adam_color, predict_color);
	Adam::setHyperParams(adam_pos, 1e-3, 0.9, 0.999, 1e-8);
	Adam::setHyperParams(adam_color, 1e-3, 0.9, 0.999, 1e-8);

	Project::init(hr_target_proj, hr_mat.mvp, target_pos, true);
	Rasterize::init(hr_target_rast, hr_target_proj, 512, 512, 1, false);
	Interpolate::init(hr_target_intr, hr_target_rast, target_color);
	Antialias::init(hr_target_aa, hr_target_rast, hr_target_proj, hr_target_intr.kernel.out, 3);

	Project::init(hr_predict_proj, hr_mat.mvp, predict_pos, true);
	Rasterize::init(hr_predict_rast, hr_predict_proj, 512, 512, 1, false);
	Interpolate::init(hr_predict_intr, hr_predict_rast, predict_color);
	Antialias::init(hr_predict_aa, hr_predict_rast, hr_predict_proj, hr_predict_intr.kernel.out, 3);

	GLbuffer::init(gl_aa_target, target_aa.kernel.out, resolution, resolution, 3, 15);
	GLbuffer::init(gl_hr_target, hr_target_aa.kernel.out, 512, 512, 3, 14);
	GLbuffer::init(gl_target, target_flt.kernel.out, resolution, resolution, 3, 13);
	GLbuffer::init(gl_aa_predict, predict_aa.kernel.out, resolution, resolution, 3, 12);
	GLbuffer::init(gl_predict, predict_flt.kernel.out, resolution, resolution, 3, 11);
	GLbuffer::init(gl_hr_predict, hr_predict_aa.kernel.out, 512, 512, 3, 10);
}

void PresetFilter::display() {
	Matrix::forward(mat);
	Matrix::forward(hr_mat);

	Project::forward(target_proj);
	Rasterize::forward(target_rast);
	Interpolate::forward(target_intr);
	Antialias::forward(target_aa);
	Filter::forward(target_flt);

	Project::forward(predict_proj);
	Rasterize::forward(predict_rast);
	Interpolate::forward(predict_intr);
	Antialias::forward(predict_aa);
	Filter::forward(predict_flt);

	MSELoss::backward(loss);
	Filter::backward(predict_flt);
	Antialias::backward(predict_aa);
	Interpolate::backward(predict_intr);
	Rasterize::backward(predict_rast);
	Project::backward(predict_proj);
	Adam::step(adam_pos);
	Adam::step(adam_color);
	AttributeGrad::clear(predict_pos);
	AttributeGrad::clear(predict_color);

	Project::forward(hr_target_proj);
	Rasterize::forward(hr_target_rast);
	Interpolate::forward(hr_target_intr);
	Antialias::forward(hr_target_aa);

	Project::forward(hr_predict_proj);
	Rasterize::forward(hr_predict_rast);
	Interpolate::forward(hr_predict_intr);
	Antialias::forward(hr_predict_aa);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(gl_aa_predict, GL_RGB32F, GL_RGB, -1.f, 0.f, -.33333333f, 1.f);
	GLbuffer::draw(gl_predict, GL_RGB32F, GL_RGB, -.33333333f, 0.f, .33333333f, 1.f);
	GLbuffer::draw(gl_hr_predict, GL_RGB32F, GL_RGB, .33333333f, 0.f, 1.f, 1.f);
	GLbuffer::draw(gl_aa_target, GL_RGB32F, GL_RGB, -1.f, -1.f, -.33333333f, 0.f);
	GLbuffer::draw(gl_target, GL_RGB32F, GL_RGB, -.33333333f, -1.f, .33333333f, 0.f);
	GLbuffer::draw(gl_hr_target, GL_RGB32F, GL_RGB, .33333333f, -1.f, 1.f, 0.f);
	glFlush();
}

void PresetFilter::update() {
	Matrix::setRandomRotation(mat);
	Matrix::addRotation(hr_mat, .1f, 0.f, 1.f, 0.f);
}