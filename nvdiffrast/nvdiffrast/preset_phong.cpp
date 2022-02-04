#include "preset.h"

#define CONSOLE_INTERVAL 10

void PresetPhong::init() {
	loss_sum = 0.f;
	CUDA_ERROR_CHECK(cudaMallocHost(&params_, 4 * sizeof(float)));
	time = 0;
	step = 0;
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
		.1f,.5f,.7f,50.f
	};
	file.open("../../log/phong_" + std::to_string(_params[0])
		+ "_" + std::to_string(_params[1])
		+ "_" + std::to_string(_params[2])
		+ "_" + std::to_string(_params[3]) + "_log.txt");
	file << "step, Ka, Kd, Ks, alpha, predict, time" << std::endl;
	int width = 512;
	int height = 512;
	Attribute::loadOBJ("../../spot_triangulated.obj", &pos, &texel, &normal);
	Texture::loadBMP("../../spot_texture.bmp", target_texture, 4);
	Matrix::init(mat);
	Matrix::setFovy(mat, 45.f);
	Matrix::setEye(mat, -2.f, 0.f, -2.f);
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, width, height, 1, true);
	Interpolate::init(intr, rast, texel);
	Project::init(pos_proj, mat.m, pos, m_pos, false);
	Project::init(normal_proj, mat.r, normal, r_normal, false);
	Texturemap::init(target_tex, rast, intr, target_texture);


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
	params_[0] = 0.f; params_[1] = 0.f; params_[2] = 0.f; params_[3] = 1.f;
	BufferGrad::init(point, 1, 3);
	Buffer::copy(point, _point);
	BufferGrad::init(intensity, 1, 3);
	Buffer::copy(intensity, _intensity);
	BufferGrad::init(params, 4, 1);
	Buffer::copy(params, params_);
	Material::init(mtr, rast, pos_proj, normal_proj, &texel, 3, target_tex.kernel.out, nullptr);
	Material::init(mtr, *(float3*)&mat.eye, point, intensity);
	Material::init(mtr, params);
	Loss::init(loss, target_mtr.kernel.out, mtr.kernel.out, mtr.grad.out, width, height, 3);

	//Optimizer::init(point_adam, point);
	//Adam::setHyperParams(point_adam, 1e-3, .9, .999, 1e-8);
	//Optimizer::init(intensity_adam, intensity);
	//Adam::setHyperParams(intensity_adam, 1e-3, .9, .999, 1e-8);
	Optimizer::init(params_adam, params);
	Adam::setHyperParams(params_adam, 1e-2, .9, .99, 1e-8);

	GLbuffer::init(target_buffer, target_mtr.kernel.out, width, height, 3);
	GLbuffer::init(buffer, mtr.kernel.out, width, height, 3);

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
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Project::forward(pos_proj);
	Project::forward(normal_proj);
	Texturemap::forward(target_tex);
	PhongMaterial::forward(mtr);
	MSELoss::backward(loss);
	PhongMaterial::backward(mtr);
	//Adam::step(point_adam);
	//Adam::step(intensity_adam);
	Adam::step(params_adam);
	Optimizer::clampParams(params_adam, 1e-3, 1e+3);
	BufferGrad::clear(point);
	BufferGrad::clear(intensity);
	BufferGrad::clear(params);
	timespec_get(&end, TIME_UTC);
	time += double(end.tv_sec - start.tv_sec) + double(end.tv_nsec - start.tv_nsec) * 1e-9;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(target_buffer, GL_RGB32F, GL_RGB,0.f, -1.f, 1.f, 1.f);
	GLbuffer::draw(buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, 0.f, 1.f );
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
		file << step << ", " << loss_sum
			<< ", " << params_[0]
			<< ", " << params_[1]
			<< ", " << params_[2]
			<< ", " << params_[3] << ", " << time / step << std::endl;
		loss_sum = 0.f;
	}
	if (step == pause[it]) {
		play = false;
		it++;
	}

	//Matrix::setEye(mat, 4 * sin(t), 0.f, 4 * cos(t));
	//Matrix::addRotation(mat, 1.f, 0.f, 1.f, 0.f);
}