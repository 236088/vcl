#include "preset.h"

#define CONSOLE_INTERVAL 10

void PresetPhong::init() {
	int resolution = 512;
	loss_sum = 0.f;
	time = 0;
	step = 0;

	//�Ɩ��p�����[�^�ݒ�
	float _point[3]{
		-2.f,2.f,-2.f,
	};
	float _intensity[3]{
		10.f,10.f,10.f,
	};
	//Phong���˃��f�� �ڕW�p�����[�^�ݒ�
	float _params[4]{
		.1f,.5f,.7f,50.f
	};
	//�o�b�t�@�̏�����
	Buffer::init(target_point, 1, 3);
	Buffer::copy(target_point, _point);
	Buffer::init(target_intensity, 1, 3);
	Buffer::copy(target_intensity, _intensity);
	Buffer::init(target_params, 4, 1);
	Buffer::copy(target_params, _params);

	//�w�K�p�����[�^�̏�����
	CUDA_ERROR_CHECK(cudaMallocHost(&params_, 4 * sizeof(float)));
	params_[0] = 0.f;
	params_[1] = 0.f;
	params_[2] = 0.f; 
	params_[3] = 1.f;
	BufferGrad::init(params, 4, 1);
	Buffer::copy(params, params_);

	//���f���̃��[�h
	Attribute::loadOBJ("../../spot_triangulated.obj", &pos, &texel, &normal);
	//�e�N�X�`���̃��[�h
	Texture::loadBMP("../../spot_texture.bmp", target_texture, 4);

	//�s��̏�����
	Matrix::init(mat);
	Matrix::setFovy(mat, 45.f);
	Matrix::setEye(mat, -2.f, 0.f, -2.f);
	//���ʃp�C�v���C���̏�����
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, resolution, resolution, 1, true);
	Interpolate::init(intr, rast, texel);
	Project::init(pos_proj, mat.m, pos, m_pos, false);
	Project::init(normal_proj, mat.r, normal, r_normal, false);
	Texturemap::init(target_tex, rast, intr, target_texture);
	//�ڕW�摜�����p�C�v���C���̏�����
	Material::init(target_mtr, rast, pos_proj, normal_proj, &texel, 3, target_tex.kernel.out);
	Material::init(target_mtr, *(float3*)&mat.eye, target_point, target_intensity);
	Material::init(target_mtr, target_params);
	//�w�K�p�C�v���C���̏�����
	Material::init(mtr, rast, pos_proj, normal_proj, &texel, 3, target_tex.kernel.out, nullptr);
	Material::init(mtr, *(float3*)&mat.eye, target_point, target_intensity);
	Material::init(mtr, params);

	//�����֐��̏�����
	Loss::init(loss, target_mtr.kernel.out, mtr.kernel.out, mtr.grad.out, resolution, resolution, 3);

	//�œK���A���S���Y���̏�����
	Optimizer::init(params_adam, params);
	Adam::setHyperParams(params_adam, 1e-2, .9, .99, 1e-8);

	//�o�̓o�b�t�@�̏�����
	GLbuffer::init(target_buffer, target_mtr.kernel.out, resolution, resolution, 3);
	GLbuffer::init(buffer, mtr.kernel.out, resolution, resolution, 3);

	//�ڕW�摜�̃����_�����O
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
	//�t�H���[�h�p�X
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Project::forward(pos_proj);
	Project::forward(normal_proj);
	Texturemap::forward(target_tex);
	PhongMaterial::forward(mtr);
	//�����֐�
	MSELoss::backward(loss);
	//�o�b�N���[�h�p�X
	PhongMaterial::backward(mtr);
	//�œK��
	Adam::step(params_adam);
	Optimizer::clampParams(params_adam, 1e-3, 1e+3);
	BufferGrad::clear(params);
	timespec_get(&end, TIME_UTC);
	time += double(end.tv_sec - start.tv_sec) + double(end.tv_nsec - start.tv_nsec) * 1e-9;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	//�� �w�K�摜
	GLbuffer::draw(buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, 0.f, 1.f );
	//�E �ڕW�摜
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