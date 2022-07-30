#include "preset.h"

#define CONSOLE_INTERVAL 10
#define ANTIALIAS_MODE

void PresetCube::init() {
	int resolution = 32;
	loss_sum = 0.f;
	error_sum = 0.f;
	time = 0;
	step = 0;

	//モデルのロード
	Attribute::loadOBJ("../../cube.obj", &target_pos, nullptr, nullptr);
	Attribute::init(target_color, target_pos, 3);
	Attribute::copy(target_color, target_pos);
	Attribute::liner(target_color, 1.f, .5f);

	//行列の初期化
	Matrix::init(mat);
	Matrix::setEye(mat, 0.f, 0.f, 3.5f);
	Matrix::setFovy(mat, 45.f);


	//目標画像生成パイプラインの初期化
	Project::init(target_proj, mat.mvp, target_pos, true);
	Rasterize::init(target_rast, target_proj, resolution, resolution, 1, false);
	Interpolate::init(target_intr,target_rast, target_color);
#ifdef ANTIALIAS_MODE
	Antialias::init(target_aa,target_rast, target_proj, target_intr.kernel.out, 3);
#endif

	//学習パラメータの初期化
	AttributeGrad::init(pos, target_pos.vboNum, target_pos.vaoNum, 3);
	Attribute::copy(pos, target_pos);
	Attribute::addRandom(pos, -.5f, .5f);
	AttributeGrad::init(color, pos, 3);
	Attribute::addRandom(color, 0.f, 1.f);

	//学習パイプラインの初期化
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, resolution, resolution, 1, false);
	Interpolate::init(intr, rast, color);
#ifdef ANTIALIAS_MODE
	Antialias::init(aa, rast, proj, intr.kernel.out, intr.grad.out, 3);
	//損失関数の初期化
	Loss::init(loss, target_aa.kernel.out, aa.kernel.out, aa.grad.out, resolution, resolution, 3);
#else
	//損失関数の初期化
	Loss::init(loss, target_intr.kernel.out, intr.kernel.out, intr.grad.out, resolution, resolution, 3);
#endif

	//最適化アルゴリズムの初期化
	Optimizer::init(adam_pos, pos);
	Optimizer::init(adam_color, color);
	Adam::setHyperParams(adam_pos, 1e-3, 0.9, 0.999, 1e-8);
	Adam::setHyperParams(adam_color, 1e-3, 0.9, 0.999, 1e-8);

	//高画質画像用
	//行列の初期化
	Matrix::init(hr_mat);
	Matrix::setEye(hr_mat, 0.f, 0.f, 3.5f);
	Matrix::setFovy(hr_mat, 45.f);

	//目標モデル
	Project::init(hr_target_proj, hr_mat.mvp, target_pos, true);
	Rasterize::init(hr_target_rast, hr_target_proj, 512, 512, 1, false);
	Interpolate::init(hr_target_intr, hr_target_rast, target_color);
	Antialias::init(hr_target_aa, hr_target_rast, hr_target_proj, hr_target_intr.kernel.out, 3);

	//学習モデル
	Project::init(hr_proj, hr_mat.mvp, pos, true);
	Rasterize::init(hr_rast, hr_proj, 512, 512, 1, false);
	Interpolate::init(hr_intr, hr_rast, color);
	Antialias::init(hr_aa, hr_rast, hr_proj, hr_intr.kernel.out, 3);

	//出力バッファの初期化
#ifdef ANTIALIAS_MODE
	GLbuffer::init(gl_target, target_aa.kernel.out, resolution, resolution, 3);
	GLbuffer::init(gl, aa.kernel.out, resolution, resolution, 3);
#else
	GLbuffer::init(gl, intr.kernel.out, resolution, resolution, 3);
	GLbuffer::init(gl_target, target_intr.kernel.out, resolution, resolution, 3);
#endif
	GLbuffer::init(gl_hr_target, hr_target_aa.kernel.out, 512, 512, 3);
	GLbuffer::init(gl_hr, hr_aa.kernel.out, 512, 512, 3);
}

void PresetCube::display() {
	//目標画像のレンダリング
	Matrix::forward(mat);
	Project::forward(target_proj);
	Rasterize::forward(target_rast);
	Interpolate::forward(target_intr);
#ifdef ANTIALIAS_MODE
	Antialias::forward(target_aa);
#endif

	struct timespec start, end;
	timespec_get(&start, TIME_UTC);
	//フォワードパス
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
#ifdef ANTIALIAS_MODE
	Antialias::forward(aa);
	//損失関数
	MSELoss::backward(loss);
	//バックワードパス
	Antialias::backward(aa);
#else
	//損失関数
	MSELoss::backward(loss);
	//バックワードパス
#endif
	Interpolate::backward(intr);
	Rasterize::backward(rast);
	Project::backward(proj);
	//最適化
	Adam::step(adam_pos);
	Adam::step(adam_color);
	AttributeGrad::clear(pos);
	AttributeGrad::clear(color);
	timespec_get(&end, TIME_UTC);
	time += double(end.tv_sec - start.tv_sec) + double(end.tv_nsec - start.tv_nsec) * 1e-9;

	//高画質画像のレンダリング
	Matrix::forward(hr_mat);
	Project::forward(hr_target_proj);
	Rasterize::forward(hr_target_rast);
	Interpolate::forward(hr_target_intr);
	Antialias::forward(hr_target_aa);

	Project::forward(hr_proj);
	Rasterize::forward(hr_rast);
	Interpolate::forward(hr_intr);
	Antialias::forward(hr_aa);

	//バッファの描画
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	//左上 学習画像
	GLbuffer::draw(gl, GL_RGB32F, GL_RGB, -1.f, 0.f, 0.f, 1.f);
	//左下 高画質学習モデル画像
	GLbuffer::draw(gl_hr, GL_RGB32F, GL_RGB, -1.f, -1.f, 0.f, 0.f);
	//右上 目標画像
	GLbuffer::draw(gl_target, GL_RGB32F, GL_RGB, 0.f, 0.f, 1.f, 1.f);
	//右下 高画質目標モデル画像
	GLbuffer::draw(gl_hr_target, GL_RGB32F, GL_RGB, 0.f, -1.f, 1.f, 0.f);
	glFlush();
}

void PresetCube::update(double dt, double t, bool& play) {
	loss_sum += Loss::loss(loss);
	error_sum += Attribute::distanceError(pos, target_pos);
	if ((++step) % CONSOLE_INTERVAL == 0) {
		loss_sum /= CONSOLE_INTERVAL;
		error_sum /= CONSOLE_INTERVAL;
		std::cout << step << ", " << loss_sum << ", " << error_sum << " time:" << time/step << std::endl;
		loss_sum = 0.f;
		error_sum = 0.f;
	}
	//次ステップの回転
	Matrix::setRandomRotation(mat);
	Matrix::addRotation(hr_mat, .1f, 0.f, 1.f, 0.f);
}