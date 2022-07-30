#include "preset.h"

#define CONSOLE_INTERVAL 10
#define MIP_MODE 8

void PresetEarth::init() {
	loss_sum = 0.f;
	error_sum = 0.f;
	time = 0;
	step = 0;

	//モデルのロード
	Attribute::loadOBJ("../../sphere.obj", &pos, &texel, nullptr);
	//テクスチャのロード
	Texture::loadBMP("../../earth-texture.bmp", target_texture, 1);

	//行列の初期化
	Matrix::init(mat);
	Matrix::setEye(mat, 0.f, 0.f, 3.5f);
	Matrix::setFovy(mat, 45.f);
	//目標画像生成パイプラインの初期化
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(target_rast, proj, 4096, 4096, 1, true);
	Interpolate::init(target_intr, target_rast, texel);
	Texturemap::init(target_tex, target_rast, target_intr, target_texture);
	Texture::init(out_tex, target_tex.kernel.out, 4096, 4096, 3, 4);

	//学習パイプラインの初期化
	Rasterize::init(rast, proj, 512, 512, 1, true);
	Interpolate::init(intr, rast, texel);
#ifdef MIP_MODE
	TextureGrad::init(texture, target_texture.width, target_texture.height, target_texture.channel, MIP_MODE);
#else
	TextureGrad::init(texture, target_texture.width, target_texture.height, target_texture.channel, 1);
#endif
	Texturemap::init(tex, rast, intr, texture);

	//損失関数の初期化
	Loss::init(loss, out_tex.texture[3], tex.kernel.out, tex.grad.out, 512, 512, 3);
	
	//最適化アルゴリズムの初期化
	Optimizer::init(adam, texture);
	Adam::setHyperParams(adam, 1e-3, 0.9, 0.99, 1e-8);

	//目標テクスチャと学習テクスチャのMSE
	Loss::init(tex_loss, target_texture.texture[0], texture.texture[0], nullptr, target_texture.width, target_texture.height, target_texture.channel);

	//出力バッファの初期化
	GLbuffer::init(gl, tex.kernel.out, 512, 512, 3);
	GLbuffer::init(gl_target, out_tex.texture[3], 512, 512, 3);
	GLbuffer::init(gl_tex, &texture.texture[0][2048 * 512 * 3], 2048, 512, 3);
	GLbuffer::init(gl_tex_target, &target_texture.texture[0][2048 * 512 * 3], 2048, 512, 3);
}

void PresetEarth::display() {
	//目標画像のレンダリング
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(target_rast);
	Interpolate::forward(target_intr);
	Texturemap::forward(target_tex);
	Texture::bilinearDownsampling(out_tex);

	struct timespec start, end;
	timespec_get(&start, TIME_UTC);
	//フォワードパス
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Texturemap::forward(tex);
	//損失関数
	MSELoss::backward(loss);
	//バックワードパス
	Texturemap::backward(tex);
#ifdef MIP_MODE
	//最適化
	TextureGrad::gradSumup(texture);
	Adam::step(adam);
	Texture::buildMIP(texture);
#else
	Adam::step(adam);
#endif
	MSELoss::textureloss(tex_loss);
	TextureGrad::clear(texture);
	timespec_get(&end, TIME_UTC);
	time += double(end.tv_sec - start.tv_sec) + double(end.tv_nsec - start.tv_nsec) * 1e-9;


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	//左上 学習画像
	GLbuffer::draw(gl, GL_RGB32F, GL_RGB, -1.f, 0.f, 0.f, 1.f);
	//左下 学習テクスチャ一部拡大画像
	GLbuffer::draw(gl_tex, GL_RGB32F, GL_RGB, 0.f, 0.f, .25f, 1.f, -1.f, -1.f, 0.f, 0.f);
	//右上 目標画像
	GLbuffer::draw(gl_target, GL_RGB32F, GL_RGB, 0.f, 0.f, 1.f, 1.f);
	//右下 目標テクスチャ一部拡大画像
	GLbuffer::draw(gl_tex_target, GL_RGB32F, GL_RGB, 0.f, 0.f, .25f, 1.f, 0.f, -1.f, 1.f, 0.f);
	glFlush();
}

void PresetEarth::update(double dt, double t, bool& play) {
	loss_sum += Loss::loss(loss);
	error_sum -= 10.0 * log10(Loss::loss(tex_loss));
	if ((++step) % CONSOLE_INTERVAL == 0) {
		loss_sum /= CONSOLE_INTERVAL;
		error_sum /= CONSOLE_INTERVAL;
		std::cout << step << ", " << loss_sum << ", " << error_sum << " time:" << time / step << std::endl;
		loss_sum = 0.f;
		error_sum = 0.f;
	}
	//次ステップの位置・回転
	Matrix::setRandomRotation(mat);
	float x = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
	float y = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
	float z = (float)rand() / (float)RAND_MAX * 48.5f + 1.5f;
	Matrix::setTranslation(mat, x, y, 0.f);
	Matrix::setEye(mat, 0.f, 0.f, z);
}