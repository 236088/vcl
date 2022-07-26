#include "preset.h"

#define CONSOLE_INTERVAL 10
#define MIP_MODE
#define DISPLAY_TEXTURE
#define DISPLAY

void PresetEarth::init() {
	loss_sum = 0.f;
	error_sum = 0.f;
	time = 0;
	step = 0;

	Attribute::loadOBJ("../../sphere.obj", &pos, &texel, nullptr);
	Matrix::init(mat);
	Matrix::setEye(mat, 0.f, 0.f, 3.5f);
	Matrix::setFovy(mat, 45.f);
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(target_rast, proj, 4096, 4096, 1, true);
	Interpolate::init(target_intr, target_rast, texel);
	Texture::loadBMP("../../earth-texture.bmp", target_texture, 1);
	Texturemap::init(target_tex, target_rast, target_intr, target_texture);
	Texture::init(out_tex, target_tex.kernel.out, 4096, 4096, 3, 4);

	Rasterize::init(rast, proj, 512, 512, 1, true);
	Interpolate::init(intr, rast, texel);
#ifdef MIP_MODE
	TextureGrad::init(texture, target_texture.width, target_texture.height, target_texture.channel, 3);
#else
	TextureGrad::init(texture, target_texture.width, target_texture.height, target_texture.channel, 1);
#endif
	Texturemap::init(tex, rast, intr, texture);
	Loss::init(loss, out_tex.texture[3], tex.kernel.out, tex.grad.out, 512, 512, 3);
	Optimizer::init(adam, texture);
	Adam::setHyperParams(adam, 1e-3, 0.9, 0.99, 1e-8);
	Loss::init(tex_loss, target_texture.texture[0], texture.texture[0], nullptr, target_texture.width, target_texture.height, target_texture.channel);

#ifdef  DISPLAY
	GLbuffer::init(gl, tex.kernel.out, 512, 512, 3);
	GLbuffer::init(gl_target, out_tex.texture[3], 512, 512, 3);
#ifdef  DISPLAY_TEXTURE
	GLbuffer::init(gl_tex, &texture.texture[0][2048 * 512 * 3], 2048, 512, 3);
	GLbuffer::init(gl_tex_target, &target_texture.texture[0][2048 * 512 * 3], 2048, 512, 3);
#endif
#endif
}

void PresetEarth::display() {
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(target_rast);
	Interpolate::forward(target_intr);
	Texturemap::forward(target_tex);
	Texture::bilinearDownsampling(out_tex);

	struct timespec start, end;
	timespec_get(&start, TIME_UTC);
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Texturemap::forward(tex);
	MSELoss::backward(loss);
	Texturemap::backward(tex);
#ifdef MIP_MODE
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

#ifdef DISPLAY
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(gl, GL_RGB32F, GL_RGB, -1.f, 0.f, 0.f, 1.f);
	GLbuffer::draw(gl_target, GL_RGB32F, GL_RGB, 0.f, 0.f, 1.f, 1.f);
#ifdef DISPLAY_TEXTURE
	GLbuffer::draw(gl_tex, GL_RGB32F, GL_RGB, 0.f, 0.f, .25f, 1.f, -1.f, -1.f, 0.f, 0.f);
	GLbuffer::draw(gl_tex_target, GL_RGB32F, GL_RGB, 0.f, 0.f, .25f, 1.f, 0.f, -1.f, 1.f, 0.f);
#endif
	glFlush();
#endif
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
	Matrix::setRandomRotation(mat);
	float x = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
	float y = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
	float z = (float)rand() / (float)RAND_MAX * 48.5f + 1.5f;
	Matrix::setTranslation(mat, x, y, 0.f);
	Matrix::setEye(mat, 0.f, 0.f, z);
}