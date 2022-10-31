
#include "preset.h"

#define CONSOLE_INTERVAL 10
#define INIT_INTERVAL 200

void PresetPose::init() {
	int resolution = 512;
	loss_sum = 0.f;
	time = 0;

	//モデルのロード
	Attribute::loadOBJ("../../models/cube.obj", &target_pos, nullptr, nullptr);
	Attribute::init(target_color, target_pos, 3);
	Attribute::copy(target_color, target_pos);
	Attribute::liner(target_color, 1.f, .5f);

	//目標画像生成パイプラインの初期化
	Rotation::init(target_rot);
	Camera::init(target_cam, target_rot, glm::vec3(0.f, 0.f, 3.5f), glm::vec3(0.f, 0.f, -1.f), glm::vec3(0.f, 1.f, 0.f), .5f, 1.f, 1.f, 10.f);
	Project::init(target_proj, target_cam.kernel.out, target_pos, true);
	Rasterize::init(target_rast, target_proj, resolution, resolution, 1, false);
	Interpolate::init(target_intr, target_rast, target_color);
	Antialias::init(target_aa, target_rast, target_proj, target_intr.kernel.out, 3);


	//学習パラメータの初期化
	AttributeGrad::init(pos, target_pos.vboNum, target_pos.vaoNum, 3);
	Attribute::copy(pos, target_pos);
	AttributeGrad::init(color, target_color.vboNum, target_color.vaoNum, 3);
	Attribute::copy(color, target_color);

	//学習パイプラインの初期化
	Rotation::init(rot);
	Camera::init(cam, rot, glm::vec3(0.f, 0.f, 3.5f), glm::vec3(0.f, 0.f, -1.f), glm::vec3(0.f, 1.f, 0.f), .5f, 1.f, 1.f, 10.f);
	Project::init(proj, cam.kernel.out, cam.grad.out, pos, true);
	Rasterize::init(rast, proj, resolution, resolution, 1, false);
	Interpolate::init(intr, rast, color);
	Antialias::init(aa, rast, proj, intr.kernel.out, intr.grad.out, 3);
	//損失関数の初期化
	Loss::init(loss, target_aa.kernel.out, aa.kernel.out, aa.grad.out, resolution, resolution, 3);

	Rotation::init(rot, 1e-2, .9f, .999f, 1e-8);
	//出力バッファの初期化
	GLbuffer::init(gl_target, target_aa.kernel.out, resolution, resolution, 3);
	GLbuffer::init(gl, aa.kernel.out, resolution, resolution, 3);

	Rotation::setRandom(rot);
}

void PresetPose::display() {
	//バッファの描画
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(gl, GL_RGB32F, GL_RGB, -1.f, -1.f, 0.f, 1.f);
	GLbuffer::draw(gl_target, GL_RGB32F, GL_RGB, 0.f, -1.f, 1.f, 1.f);
	glFlush();
}

void PresetPose::update(double dt, double t, unsigned int step, bool& play) {
	if (step % INIT_INTERVAL == 0) {
		//目標画像のレンダリング
		Rotation::setRandom(target_rot);
		Rotation::forward(target_rot);
		Camera::forward(target_cam);
		Project::forward(target_proj);
		Rasterize::forward(target_rast);
		Interpolate::forward(target_intr);
		Antialias::forward(target_aa);

		Rotation::reset(rot);
		Rotation::setRandom(rot);
	}

	struct timespec start, end;
	timespec_get(&start, TIME_UTC);

	Rotation::forward(rot);
	Camera::forward(cam);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Antialias::forward(aa);

	SSIMLoss::backward(loss);

	Antialias::backward(aa);
	Interpolate::backward(intr);
	Rasterize::backward(rast);
	Project::backward(proj);
	Camera::backward(cam);
	Rotation::backward(rot);

	Rotation::step(rot);

	timespec_get(&end, TIME_UTC);
	time += double(end.tv_sec - start.tv_sec) + double(end.tv_nsec - start.tv_nsec) * 1e-9;

	loss_sum += Loss::loss(loss);
	if (step % CONSOLE_INTERVAL == 0) {
		loss_sum /= CONSOLE_INTERVAL;
		std::cout << loss_sum  << " time:" << time / step << std::endl;
		loss_sum = 0.f;
	}
}