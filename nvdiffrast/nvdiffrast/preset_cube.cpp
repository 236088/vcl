#include "preset.h"

#define CONSOLE_INTERVAL 10
#define ANTIALIAS_MODE
#define DISPLAY_HIGH_RESOLUTION

void PresetCube::init() {
	int resolution = 4;
	loss_sum = 0.f;
	error_sum = 0.f;
	time = 0;
	step = 0;
	file.open("../../cube_log_"+std::to_string(resolution)+".txt");
	//file.open("F:/vcl/picture/cube/cube_log_"+std::to_string(resolution)+".txt");
	file << "step, loss, error" << std::endl;

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
#ifdef ANTIALIAS_MODE
	Antialias::init(target_aa,target_rast, target_proj, target_intr.kernel.out, 3);
	GLbuffer::init(gl_target, target_aa.kernel.out, resolution, resolution, 3);
#else
	GLbuffer::init(gl_target, target_intr.kernel.out, resolution, resolution, 3);
#endif

	AttributeGrad::init(pos, target_pos.vboNum, target_pos.vaoNum, 3);
	Attribute::copy(pos, target_pos);
	Attribute::addRandom(pos, -.5f, .5f);
	AttributeGrad::init(color, pos, 3);
	Attribute::addRandom(color, 0.f, 1.f);

	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, resolution, resolution, 1, false);
	Interpolate::init(intr, rast, color);
#ifdef ANTIALIAS_MODE
	Antialias::init(aa, rast, proj, intr.kernel.out, intr.grad.out, 3);
	Loss::init(loss, target_aa.kernel.out, aa.kernel.out, aa.grad.out, resolution, resolution, 3);
	GLbuffer::init(gl, aa.kernel.out, resolution, resolution, 3);
#else
	Loss::init(loss, target_intr.kernel.out, intr.kernel.out, intr.grad.out, resolution, resolution, 3);
	GLbuffer::init(gl, intr.kernel.out, resolution, resolution, 3);
#endif

	Optimizer::init(adam_pos, pos);
	Optimizer::init(adam_color, color);
	Adam::setHyperParams(adam_pos, 1e-3, 0.9, 0.999, 1e-8);
	Adam::setHyperParams(adam_color, 1e-3, 0.9, 0.999, 1e-8);


#ifdef DISPLAY_HIGH_RESOLUTION
	Project::init(hr_target_proj, mat.mvp, target_pos, true);
	Rasterize::init(hr_target_rast, hr_target_proj, 512, 512, 1, false);
	Interpolate::init(hr_target_intr, hr_target_rast, target_color);
	Antialias::init(hr_target_aa, hr_target_rast, hr_target_proj, hr_target_intr.kernel.out, 3);
	GLbuffer::init(gl_hr_target, hr_target_aa.kernel.out, 512, 512, 3);

	Project::init(hr_proj, mat.mvp, pos, true);
	Rasterize::wireframeinit(hr_rast, hr_proj, 512, 512);
	GLbuffer::init(gl_hr, hr_rast.kernel.out, 512, 512, 4);
#endif
}

void PresetCube::display() {
	Matrix::forward(mat);

	Project::forward(target_proj);
	Rasterize::forward(target_rast);
	Interpolate::forward(target_intr);
#ifdef ANTIALIAS_MODE
	Antialias::forward(target_aa);
#endif

	struct timespec start, end;
	timespec_get(&start, TIME_UTC);
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
#ifdef ANTIALIAS_MODE
	Antialias::forward(aa);
	MSELoss::backward(loss);
	Antialias::backward(aa);
#else
	MSELoss::backward(loss);
#endif
	Interpolate::backward(intr);
	Rasterize::backward(rast);
	Project::backward(proj);
	Adam::step(adam_pos);
	Adam::step(adam_color);
	AttributeGrad::clear(pos);
	AttributeGrad::clear(color);
	timespec_get(&end, TIME_UTC);
	time += double(end.tv_sec - start.tv_sec) + double(end.tv_nsec - start.tv_nsec) * 1e-9;


#ifdef DISPLAY_HIGH_RESOLUTION
	Project::forward(hr_target_proj);
	Rasterize::forward(hr_target_rast);
	Interpolate::forward(hr_target_intr);
	Antialias::forward(hr_target_aa);

	Project::forward(hr_proj);
	Rasterize::drawforward(hr_rast);
#endif

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(gl_target, GL_RGB32F, GL_RGB, -1.f, 0.f, 0.f, 1.f);
	GLbuffer::draw(gl, GL_RGB32F, GL_RGB, 0.f, 0.f, 1.f, 1.f);
#ifdef DISPLAY_HIGH_RESOLUTION
	GLbuffer::draw(gl_hr_target, GL_RGB32F, GL_RGB, -1.f, -1.f, 0.f, 0.f);
	GLbuffer::draw(gl_hr, GL_RGBA32F, GL_RGBA, 0.f, -1.f, 1.f, 0.f);
#endif
	glFlush();
}

void PresetCube::update(double dt, double t, bool& play) {
	loss_sum += Loss::loss(loss);
	error_sum += Attribute::distanceError(pos, target_pos);
	if ((++step) % CONSOLE_INTERVAL == 0) {
		loss_sum /= CONSOLE_INTERVAL;
		error_sum /= CONSOLE_INTERVAL;
		std::cout << step << "," << loss_sum << "," << error_sum << "time:" << time << std::endl;
		file << step << "," << loss_sum << "," << error_sum  << std::endl;
		loss_sum = 0.f;
		error_sum = 0.f;
	}
	if (step == pause[it]) {
		play = false;
		it++;
	}

	Matrix::setRandomRotation(mat);
}