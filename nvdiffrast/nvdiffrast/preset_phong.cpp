#include "preset.h"

#define CONSOLE_INTERVAL 10

void PresetPhong::init() {
	int resolution = 512;
	loss_sum = 0.f;
	time = 0;
	step = 0;

	//照明パラメータ設定
	float _point[3]{
		-2.f,2.f,-2.f,
	};
	float _intensity[3]{
		10.f,10.f,10.f,
	};
	//Phong反射モデル 目標パラメータ設定
	float _params[4]{
		.1f,.5f,.7f,50.f
	};
	//バッファの初期化
	Buffer::init(target_point, 1, 3);
	Buffer::copy(target_point, _point);
	Buffer::init(target_intensity, 1, 3);
	Buffer::copy(target_intensity, _intensity);
	Buffer::init(target_params, 4, 1);
	Buffer::copy(target_params, _params);

	//学習パラメータの初期化
	CUDA_ERROR_CHECK(cudaMallocHost(&params_, 4 * sizeof(float)));
	params_[0] = 0.f;
	params_[1] = 0.f;
	params_[2] = 0.f; 
	params_[3] = 1.f;
	BufferGrad::init(params, 4, 1);
	Buffer::copy(params, params_);

	//モデルのロード
	Attribute::loadOBJ("../../spot_triangulated.obj", &pos, &texel, &normal);
	//テクスチャのロード
	Texture::loadBMP("../../spot_texture.bmp", target_texture, 4);

	//行列の初期化
	Matrix::init(mat);
	Matrix::setFovy(mat, 45.f);
	Matrix::setEye(mat, -2.f, 0.f, -2.f);
	//共通パイプラインの初期化
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, resolution, resolution, 1, true);
	Interpolate::init(intr, rast, texel);
	Project::init(pos_proj, mat.m, pos, m_pos, false);
	Project::init(normal_proj, mat.r, normal, r_normal, false);
	Texturemap::init(target_tex, rast, intr, target_texture);
	//目標画像生成パイプラインの初期化
	Material::init(target_mtr, rast, pos_proj, normal_proj, &texel, 3, target_tex.kernel.out);
	Material::init(target_mtr, *(float3*)&mat.eye, target_point, target_intensity);
	Material::init(target_mtr, target_params);
	//学習パイプラインの初期化
	Material::init(mtr, rast, pos_proj, normal_proj, &texel, 3, target_tex.kernel.out, nullptr);
	Material::init(mtr, *(float3*)&mat.eye, target_point, target_intensity);
	Material::init(mtr, params);

	//損失関数の初期化
	Loss::init(loss, target_mtr.kernel.out, mtr.kernel.out, mtr.grad.out, resolution, resolution, 3);

	//最適化アルゴリズムの初期化
	Optimizer::init(params_adam, params);
	Adam::setHyperParams(params_adam, 1e-2, .9, .99, 1e-8);

	//出力バッファの初期化
	GLbuffer::init(target_buffer, target_mtr.kernel.out, resolution, resolution, 3);
	GLbuffer::init(buffer, mtr.kernel.out, resolution, resolution, 3);

	//目標画像のレンダリング
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Project::forward(pos_proj);
	Project::forward(normal_proj);
	Texturemap::forward(target_tex);
	PhongMaterial::forward(target_mtr);
}

void PresetPhong::display(void) {
	struct timespec start, end;
	timespec_get(&start, TIME_UTC);
	//フォワードパス
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Project::forward(pos_proj);
	Project::forward(normal_proj);
	Texturemap::forward(target_tex);
	PhongMaterial::forward(mtr);
	//損失関数
	MSELoss::backward(loss);
	//バックワードパス
	PhongMaterial::backward(mtr);
	//最適化
	Adam::step(params_adam);
	Optimizer::clampParams(params_adam, 1e-3, 1e+3);
	BufferGrad::clear(params);
	timespec_get(&end, TIME_UTC);
	time += double(end.tv_sec - start.tv_sec) + double(end.tv_nsec - start.tv_nsec) * 1e-9;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	//左 学習画像
	GLbuffer::draw(buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, 0.f, 1.f );
	//右 目標画像
	GLbuffer::draw(target_buffer, GL_RGB32F, GL_RGB,0.f, -1.f, 1.f, 1.f);
	glFlush();
}

void PresetPhong::update(double dt, double t, bool& play) {
	loss_sum += Loss::loss(loss);
	if ((++step) % CONSOLE_INTERVAL == 0) {
		Buffer::copy(params_, params);
		loss_sum /= CONSOLE_INTERVAL;
		std::cout << step << ", " << loss_sum
			<< ", " << params_[0]
			<< ", " << params_[1]
			<< ", " << params_[2]
			<< ", " << params_[3] << " time:" << time / step << std::endl;
		loss_sum = 0.f;
	}
}