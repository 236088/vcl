#include "preset.h"

#define CONSOLE_INTERVAL 10
#define EXIT_COUNT 5000

void PresetPhong::init() {
	loss_sum = 0.f;
	step = 0;
	file.open("../../phong_log.txt");
	file << "step, predict" << std::endl;
	int width = 512;
	int height = 512;
	Attribute::loadOBJ("../../spot_triangulated.obj", &pos, &texel, nullptr);
	Texture::loadBMP("../../spot_texture.bmp", target_texture, 4);
	TextureGrad::init(predict_texture, target_texture.width, target_texture.height, target_texture.channel, 4);
	float color[3] = { 1.f,1.f,1.f };
	Texture::setColor(predict_texture, color);
	Matrix::init(mat);
	Matrix::setRotation(mat, 180.f, 0.f, 1.f, 0.f);
	Matrix::setFovy(mat, 30.f);
	Matrix::setEye(mat, 0.f, 0.f, 4.f);
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, width, height, 1, true);
	Interpolate::init(intr, rast, texel);
	Project::init(pos_proj, mat.m, pos, m_pos, false);
	Normalcalc::init(norm, pos, normal);
	Project::init(normal_proj, mat.r, normal, r_normal, false);
	Texturemap::init(target_tex, rast, intr, target_texture);
	Material::init(target_mtr, rast, pos_proj, normal_proj, &texel, 3, target_tex.kernel.out, nullptr, nullptr, nullptr);
	float3 point[4]{
		-2.f, -2.f, -5.f,
		0.f, -3.f, -5.f,
		3.f, 0.f, -5.f,
		0.f, 3.f, -5.f,
	};
	float lightintensity[12]{
		1.f,1.f,1.f,
		1.f,1.f,1.f,
		.5f,.5f,.5f,
		.25f,.25f,.25f,
	};
	PhongMaterial::init(target_mtr, *(float3*)&mat.eye, 1, point, lightintensity, .1f, .7f, .5f, 5.f);
	Texturemap::init(predict_tex, rast, intr, predict_texture);
	Material::init(predict_mtr, rast, pos_proj, normal_proj, &texel, 3, target_tex.kernel.out, predict_tex.grad.out);
	PhongMaterial::init(predict_mtr, *(float3*)&mat.eye, 1, point, lightintensity, 0.f, 0.f, 0.f, 1.f);
	Loss::init(loss, target_mtr.kernel.out, predict_mtr.kernel.out, predict_mtr.grad.out, width, height, 3);
	
	Optimizer::init(mtr_adam, predict_mtr.kernel.params, predict_mtr.grad.params, 4, 4, 1, 1);
	Adam::setHyperParams(mtr_adam, 1e-2, .9, .999, 1e-8);
	//Optimizer::randomParams(mtr_adam, 1e-3, 1.f);
	//Optimizer::init(tex_adam, predict_texture);
	//Adam::setHyperParams(tex_adam, 1e-3, .9, .999, 1e-8);

	//Texture::init(white, predict_texture.width, predict_texture.height, predict_texture.channel, 1);
	//Texture::setColor(white, color);
	//Texturemap::init(white_tex, rast, intr, white);
	//Material::init(white_mtr, rast, pos_proj, normal_proj, &texel, 3, white_tex.kernel.out);
	//PhongMaterial::init(white_mtr, *(float3*)&mat.eye, 1, point, lightintensity, 0.f, 0.f, 0.f, 0.f);
	//cudaFree(white_mtr.kernel.params);
	//white_mtr.kernel.params = predict_mtr.kernel.params;

	GLbuffer::init(target_buffer, target_mtr.kernel.out, width, height, 3);
	GLbuffer::init(predict_buffer, predict_mtr.kernel.out, width, height, 3);
	//GLbuffer::init(white_buffer, white_mtr.kernel.out, width, height, 3);
	//GLbuffer::init(tex_buffer, predict_texture.texture[0], predict_texture.width,predict_texture.height, 3);
}

void PresetPhong::display(void) {
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Project::forward(pos_proj);
	Normalcalc::forward(norm);
	Project::forward(normal_proj);
	Texturemap::forward(target_tex);
	PhongMaterial::forward(target_mtr);

	//Texturemap::forward(predict_tex);
	PhongMaterial::forward(predict_mtr);
	MSELoss::backward(loss);
	PhongMaterial::backward(predict_mtr);
	//Texturemap::backward(predict_tex);

	//Texturemap::forward(white_tex);
	//PhongMaterial::forward(white_mtr);

	//TextureGrad::gradSumup(predict_texture);
	//Adam::step(tex_adam);
	//Texture::buildMIP(predict_texture);
	//TextureGrad::clear(predict_texture);
	Adam::step(mtr_adam);
	Material::clear(predict_mtr);


	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(target_buffer, GL_RGB32F, GL_RGB,0.f, -1.f, 1.f, 1.f);
	GLbuffer::draw(predict_buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, 0.f, 1.f );
	//GLbuffer::draw(white_buffer, GL_RGB32F, GL_RGB,0.f, -1.f, 1.f, 0.f);
	//GLbuffer::draw(tex_buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, 0.f, 0.f );
	glFlush();
}

void PresetPhong::update(double dt) {
	loss_sum += Loss::loss(loss);
	if ((++step) % CONSOLE_INTERVAL == 0) {
		std::cout << step << "," << loss_sum / CONSOLE_INTERVAL << std::endl;
		file << step << "," << loss_sum / CONSOLE_INTERVAL << std::endl;
		loss_sum = 0.f;
	}
	if (step % EXIT_COUNT == 0) {
		file.close();
		exit(0);
	}
	Matrix::addRotation(mat, .25f, 0.f, 1.f, 0.f);
}