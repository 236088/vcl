#include "preset.h"

void PresetPBR::init() {
	batch = 0;
	int width = 512;
	int height = 512;
	t = 0.f;
	float3 point[4]{
		0.f, -2.f, -1.f,
		0.f, -3.f, -5.f,
		3.f, 0.f, -5.f,
		0.f, 3.f, -5.f,
	};
	float lightintensity[12]{
		2.f,2.f,2.f,
		1.f,1.f,1.f,
		.5f,.5f,.5f,
		.25f,.25f,.25f,
	};
	float rough[1] = {.5f };
	float nor[3] = { 0.f,0.f,1.f };
	Attribute::loadOBJ("../../simple_sphere.obj", &pos, &texel, nullptr);
	Texture::loadBMP("../../Tiles074_1K_Color.bmp", target_diff, 1);
	Texture::loadBMP("../../Tiles074_1K_Roughness.bmp", target_rough, 1);
	Texture::loadBMP("../../Tiles074_1K_NormalGL.bmp", target_nor, 1);
	Texture::liner(target_nor, 2.f, -1.f);
	Texture::normalize(target_nor);
	Texture::loadBMP("../../Tiles074_1K_Displacement.bmp", target_disp, 1);
	Matrix::init(mat);
	Matrix::setEye(mat, 0.f, 2.f, 4.f);
	Matrix::setFovy(mat, 30.f);
	Project::init(proj, mat.mvp, pos, true);
	Rasterize::init(rast, proj, width, height, 1, true);
	Interpolate::init(intr, rast, texel);
	Project::init(pos_proj, mat.m, pos, m_pos, false);
	Normalcalc::init(norm, pos, normal);
	Project::init(normal_proj, mat.r, normal, r_normal, false);
	Texturemap::init(target_diff_tex, rast, intr, target_diff);
	Texturemap::init(target_rough_tex, rast, intr, target_rough);
	Texturemap::init(target_nor_tex, rast, intr, target_nor);
	Texturemap::init(target_disp_tex, rast, intr, target_disp);
	Material::init(target_mtr, rast, pos_proj, normal_proj, &texel, 3,
		target_diff_tex.kernel.out, target_rough_tex.kernel.out, target_nor_tex.kernel.out, target_disp_tex.kernel.out);
	//PBRMaterial::init(target_mtr, *(float3*)&mat.eye, 1, point, lightintensity, 2.19f);

	TextureGrad::init(predict_diff, target_diff.width, target_diff.height, target_diff.channel, 1);
	TextureGrad::init(predict_rough, target_rough.width, target_rough.height, target_rough.channel, 1);
	Texture::setColor(predict_rough, rough);
	TextureGrad::init(predict_nor, target_nor.width, target_nor.height, target_nor.channel, 1);
	Texture::setColor(predict_nor, nor);
	TextureGrad::init(predict_disp, target_disp.width, target_disp.height, target_disp.channel, 1);
	Texturemap::init(predict_diff_tex, rast, intr, predict_diff);
	Texturemap::init(predict_rough_tex, rast, intr, predict_rough);
	Texturemap::init(predict_nor_tex, rast, intr, predict_nor);
	Texturemap::init(predict_disp_tex, rast, intr, predict_disp);
	Material::init(predict_mtr, rast, pos_proj, normal_proj, &texel, 3,
		predict_diff_tex.kernel.out, predict_rough_tex.kernel.out, predict_nor_tex.kernel.out, predict_disp_tex.kernel.out,
		predict_diff_tex.grad.out, predict_rough_tex.grad.out, predict_nor_tex.grad.out, predict_disp_tex.grad.out);
	//PBRMaterial::init(predict_mtr, *(float3*)&mat.eye, 1, point, lightintensity, 2.19f);
	Loss::init(loss, target_mtr.kernel.out, predict_mtr.kernel.out, predict_mtr.grad.out, width, height, 3);
	Optimizer::init(diff_adam, predict_diff);
	Optimizer::init(rough_adam, predict_rough);
	Optimizer::init(nor_adam, predict_nor);
	Adam::setHyperParams(diff_adam, 1e-2, 0.f, 0.f, 1e-8);
	Adam::setHyperParams(rough_adam, 1e-2, 0.f, 0.f, 1e-8);
	Adam::setHyperParams(nor_adam, 1e-2, 0.f, 0.f, 1e-8);

	GLbuffer::init(target_diff_buffer, target_diff.texture[0], target_diff.width, target_diff.height, 3);
	GLbuffer::init(target_rough_buffer, target_rough.texture[0], target_rough.width, target_rough.height, 1);
	GLbuffer::init(target_nor_buffer, target_nor.texture[0], target_nor.width, target_nor.height, 3);
	GLbuffer::init(target_mtr_buffer, target_mtr.kernel.out, width, height, 3);
	GLbuffer::init(predict_diff_buffer, predict_diff.texture[0], predict_diff.width, predict_diff.height, 3);
	GLbuffer::init(predict_rough_buffer, predict_rough.texture[0], predict_rough.width, predict_rough.height, 1);
	GLbuffer::init(predict_nor_buffer, predict_nor.texture[0], predict_nor.width, predict_nor.height, 3);
	GLbuffer::init(predict_mtr_buffer, predict_mtr.kernel.out, width, height, 3);
}

void PresetPBR::display(void) {
	Matrix::forward(mat);
	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Project::forward(pos_proj);
	Normalcalc::forward(norm);
	Project::forward(normal_proj);
	Texturemap::forward(target_diff_tex);
	Texturemap::forward(target_rough_tex);
	Texturemap::forward(target_nor_tex);
	Texturemap::forward(target_disp_tex);
	PBRMaterial::forward(target_mtr);

	Texturemap::forward(predict_diff_tex);
	Texturemap::forward(predict_rough_tex);
	Texturemap::forward(predict_nor_tex);
	Texturemap::forward(predict_disp_tex);
	PBRMaterial::forward(predict_mtr);

	MSELoss::backward(loss);
	PBRMaterial::backward(predict_mtr);
	Texturemap::backward(predict_diff_tex);
	Texturemap::backward(predict_rough_tex);
	Texturemap::backward(predict_nor_tex);

	TextureGrad::gradSumup(predict_diff);
	Adam::step(diff_adam);
	TextureGrad::clear(predict_diff);
	TextureGrad::buildMIP(predict_diff);

	TextureGrad::gradSumup(predict_rough);
	Adam::step(rough_adam);
	TextureGrad::clear(predict_rough);
	Texture::clamp(predict_rough, 1e-3, 1.f);
	TextureGrad::buildMIP(predict_rough);

	TextureGrad::gradSumup(predict_nor);
	Adam::step(nor_adam);
	TextureGrad::clear(predict_nor);
	TextureGrad::buildMIP(predict_nor);
	Texture::normalize(predict_nor);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(predict_nor_buffer, GL_RGB32F, GL_RGB, -1.f, 0.f, -.5f, 1.f);
	GLbuffer::draw(predict_rough_buffer, GL_LUMINANCE, GL_RED, -.5f, 0.f, 0.f, 1.f);
	GLbuffer::draw(predict_diff_buffer, GL_RGB32F, GL_RGB, 0.f, 0.f, .5f, 1.f);
	GLbuffer::draw(predict_mtr_buffer, GL_RGB32F, GL_RGB, .5f, 0.f, 1.f, 1.f);
	GLbuffer::draw(target_nor_buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, -.5f, 0.f);
	GLbuffer::draw(target_rough_buffer, GL_LUMINANCE, GL_RED, -.5f, -1.f, 0.f, 0.f);
	GLbuffer::draw(target_diff_buffer, GL_RGB32F, GL_RGB, 0.f, -1.f, .5f, 0.f);
	GLbuffer::draw(target_mtr_buffer, GL_RGB32F, GL_RGB, .5f, -1.f, 1.f, 0.f);
	glFlush();
}

void PresetPBR::update(double dt, double t, bool& play) {
	Matrix::setRandomRotation(mat);
}