#include "preset.h"

#define CONSOLE_INTERVAL 10

void PresetPhong::init() {
	loss_sum = 0.f;
	step = 0;
	t = 0;
	file.open("../../phong_log.txt");
	//file.open("F:/vcl/picture/phong/phong_log.txt");
	file << "step, predict" << std::endl;
	int width = 512;
	int height = 512;
	Attribute::loadOBJ("../../spot_triangulated.obj", &pos, &texel, &normal);
	Texture::loadBMP("../../spot_texture.bmp", target_texture, 4);
	Matrix::init(mat);
	Matrix::setFovy(mat, 30.f);
	Matrix::setEye(mat, -3.f, 0.f, -3.f);
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, width, height, 1, true);
	Interpolate::init(intr, rast, texel);
	Project::init(pos_proj, mat.m, pos, m_pos, false);
	Project::init(normal_proj, mat.r, normal, r_normal, false);
	Texturemap::init(target_tex, rast, intr, target_texture);

	float _point[12]{
		-2.f,2.f,-2.f,
		2.f,2.f,2.f,
		2.f,-2.f,-2.f,
		-2.f,-2.f,2.f,
	};
	float _intensity[12]{
		10.f,10.f,10.f,
		1.f,1.f,1.f,
		1.f,1.f,1.f,
		1.f,1.f,1.f,
	};
	float _params[4]{
		.1f,.7f,.5f,  5.f
	};
	Buffer::init(target_point, 1, 3);
	Buffer::copy(target_point, _point);
	Buffer::init(target_intensity, 1, 3);
	Buffer::copy(target_intensity, _intensity);
	Buffer::init(target_params, 4, 1);
	Buffer::copy(target_params, _params);
	Material::init(target_mtr, rast, pos_proj, normal_proj, &texel, 3, target_tex.kernel.out);
	Material::init(target_mtr, *(float3*)&mat.eye, target_point, target_intensity);
	Material::init(target_mtr, target_params);
	float point_[12]{
		0.f,0.f,1.f,
		1.f,-1.f,1.f,
		-1.f,1.f,1.f,
		-1.f,-1.f,1.f,
	};
	float intensity_[12]{
		0.f,0.f,0.f,
		0.f,0.f,0.f,
		0.f,0.f,0.f,
		0.f,0.f,0.f,
	};
	float params_[4]{
		.0f, .0f, .0f, 1.f
	};
	BufferGrad::init(predict_point, 1, 3);
	Buffer::copy(predict_point, _point);
	BufferGrad::init(predict_intensity, 1, 3);
	Buffer::copy(predict_intensity, _intensity);
	BufferGrad::init(predict_params, 4, 1);
	Buffer::copy(predict_params, params_);
	Material::init(predict_mtr, rast, pos_proj, normal_proj, &texel, 3, target_tex.kernel.out, nullptr);
	Material::init(predict_mtr, *(float3*)&mat.eye, predict_point, predict_intensity);
	Material::init(predict_mtr, predict_params);
	Loss::init(loss, target_mtr.kernel.out, predict_mtr.kernel.out, predict_mtr.grad.out, width, height, 3);

	//Optimizer::init(point_adam, predict_point);
	//Adam::setHyperParams(point_adam, 1e-3, .9, .999, 1e-8);
	//Optimizer::init(intensity_adam, predict_intensity);
	//Adam::setHyperParams(intensity_adam, 1e-3, .9, .999, 1e-8);
	Optimizer::init(params_adam, predict_params);
	Adam::setHyperParams(params_adam, 1e-2, .9, .999, 1e-8);

	GLbuffer::init(target_buffer, target_mtr.kernel.out, width, height, 3);
	GLbuffer::init(predict_buffer, predict_mtr.kernel.out, width, height, 3);

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
	BufferGrad::clear(predict_point);
	BufferGrad::clear(predict_intensity);
	BufferGrad::clear(predict_params);
	PhongMaterial::forward(predict_mtr);
	MSELoss::backward(loss);
	PhongMaterial::backward(predict_mtr);
	//Adam::step(point_adam);
	//Adam::step(intensity_adam);
	Adam::step(params_adam);
	Optimizer::clampParams(params_adam, 1e-3, 1e+3);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(target_buffer, GL_RGB32F, GL_RGB,0.f, -1.f, 1.f, 1.f);
	GLbuffer::draw(predict_buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, 0.f, 1.f );
	glFlush();
}

void PresetPhong::update(double dt, double t, bool& play) {
	loss_sum += Loss::loss(loss);
	t += dt;
	if ((++step) % CONSOLE_INTERVAL == 0) {
		std::cout << step << "," << loss_sum / CONSOLE_INTERVAL << std::endl;
		file << step << "," << loss_sum / CONSOLE_INTERVAL << std::endl;
		loss_sum = 0.f;
	}
	if (step == pause[5]) {
		exit(0);
		play = false;
		it++;
	}

	//Matrix::setEye(mat, 4 * sin(t), 0.f, 4 * cos(t));
	//Matrix::addRotation(mat, 1.f, 0.f, 1.f, 0.f);
}