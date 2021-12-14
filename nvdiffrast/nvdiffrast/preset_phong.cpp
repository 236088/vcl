#include "preset.h"

void PresetPhong::init() {
	int width = 512;
	int height = 512;
	Attribute::loadOBJ("../../spot_triangulated.obj", &pos, &texel, nullptr);
	Texture::loadBMP("../../spot_texture.bmp", target_texture, 4);
	TextureGrad::init(predict_texture, target_texture.width, target_texture.height, target_texture.channel, 4);
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
	float3 direction[4]{
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
	PhongMaterial::init(target_mtr, *(float3*)&mat.eye, 1, direction, lightintensity, .2f, .7f, .5f, 3.f);
	Antialias::init(target_aa, rast, proj,  target_mtr.kernel.out, 3);
	Texturemap::init(predict_tex, rast, intr, predict_texture);
	Material::init(predict_mtr, rast, pos_proj, normal_proj, &texel, 3, predict_tex.kernel.out, predict_tex.grad.out);
	PhongMaterial::init(predict_mtr, *(float3*)&mat.eye, 1, direction, lightintensity, 0.f, 0.f, 0.f, 0.f);
	Antialias::init(predict_aa, rast, proj, predict_mtr.kernel.out, predict_mtr.grad.out, 3);
	Loss::init(loss, target_aa.kernel.out, predict_aa.kernel.out, predict_aa.grad.out, width, height, 3);
	
	Optimizer::init(mtr_adam, predict_mtr.kernel.params, predict_mtr.grad.params, 4, 4, 1, 1);
	Adam::setHyperParams(mtr_adam, 1e-3, .9, .999, 1e-8);
	Optimizer::init(tex_adam, predict_texture);
	Optimizer::randomParams(mtr_adam, 1e-3, 1.f);
	Adam::setHyperParams(tex_adam, 1e-3, .9, .999, 1e-8);

	GLbuffer::init(target_buffer, target_aa.kernel.out, width, height, 3, 15);
	GLbuffer::init(predict_buffer, predict_aa.kernel.out, width, height, 3, 14);
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
	Antialias::forward(target_aa);

	Texturemap::forward(predict_tex);
	PhongMaterial::forward(predict_mtr);
	Antialias::forward(predict_aa);
	MSELoss::backward(loss);
	Antialias::backward(predict_aa);
	PhongMaterial::backward(predict_mtr);
	Texturemap::backward(predict_tex);

	TextureGrad::gradSumup(predict_texture);
	Adam::step(mtr_adam);
	Adam::step(tex_adam);
	Texture::buildMIP(predict_texture);
	Material::clear(predict_mtr);
	TextureGrad::clear(predict_texture);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(target_buffer, GL_RGB32F, GL_RGB,0.f, -1.f, 1.f, 1.f);
	GLbuffer::draw(predict_buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, 0.f, 1.f );
	glFlush();
}

void PresetPhong::update(void) {
	Matrix::addRotation(mat, .25f, 0.f, 1.f, 0.f);
}