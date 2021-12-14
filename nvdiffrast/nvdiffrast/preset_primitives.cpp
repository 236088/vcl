#include "preset.h"

void PresetPrimitives::init() {
	int width = 512;
	int height = 512;
	t = 0.f;
	float3 direction[4]{
		0.f, -1.f,0.f,
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
	float diff[3] = { .5f,.5f,.5f };
	float rough[1] = { .5f };
	float nor[3] = { 0.f,0.f,1.f };
	Attribute::loadOBJ("../../simple_sphere.obj", &target_pos, &target_texel, nullptr);
	Texture::loadBMP("../../Bricks059_1K_Color.bmp", target_diff, 4);
	Texture::loadBMP("../../Bricks059_1K_Roughness.bmp", target_rough, 4);
	Texture::loadBMP("../../Bricks059_1K_NormalGL.bmp", target_nor, 4);
	Texture::liner(target_nor, 2.f, -1.f);
	Texture::normalize(target_nor);
	Texture::loadBMP("../../Bricks059_1K_Displacement.bmp", target_disp, 4);
	Matrix::init(mat);
	Matrix::setEye(mat, 0.f, 2.f, 4.f);
	Matrix::setFovy(mat, 30.f);
	Project::init(target_proj, mat.mvp, target_pos, true);
	Rasterize::init(target_rast, target_proj, width, height, 1, true);
	Interpolate::init(target_intr, target_rast, target_texel);
	Project::init(target_pos_proj, mat.m, target_pos, target_m_pos, false);
	Normalcalc::init(target_norm, target_pos, target_normal);
	Project::init(target_normal_proj, mat.r, target_normal, target_r_normal, false);
	Texturemap::init(target_diff_tex, target_rast, target_intr, target_diff);
	Texturemap::init(target_rough_tex, target_rast, target_intr, target_rough);
	Texturemap::init(target_nor_tex, target_rast, target_intr, target_nor);
	Texturemap::init(target_disp_tex, target_rast, target_intr, target_disp);
	Material::init(target_mtr, target_rast, target_pos_proj, target_normal_proj, &target_texel, 3,
		target_diff_tex.kernel.out, target_rough_tex.kernel.out, target_nor_tex.kernel.out, target_disp_tex.kernel.out);
	PBRMaterial::init(target_mtr, *(float3*)&mat.eye, 1, direction, lightintensity, 2.19f);

	Attribute::loadOBJ("../../sphere.obj", &predict_pos, &predict_texel, nullptr);
	TextureGrad::init(predict_diff, 2048, 1536, target_diff.channel, 4);
	Texture::setColor(predict_diff, diff);
	TextureGrad::init(predict_rough, 2048, 1536, target_rough.channel, 4);
	Texture::setColor(predict_rough, rough);
	TextureGrad::init(predict_nor, 2048, 1536, target_nor.channel, 4);
	Texture::setColor(predict_nor, nor);
	TextureGrad::init(predict_disp, 2048, 1536, target_disp.channel, 4);
	Project::init(predict_proj, mat.mvp, predict_pos, true);
	Rasterize::init(predict_rast, predict_proj, width, height, 1, true);
	Interpolate::init(predict_intr, predict_rast, predict_texel);
	Project::init(predict_pos_proj, mat.m, predict_pos, predict_m_pos, false);
	Normalcalc::init(predict_norm, predict_pos, predict_normal);
	Project::init(predict_normal_proj, mat.r, predict_normal, predict_r_normal, false);
	Texturemap::init(predict_diff_tex, predict_rast, predict_intr, predict_diff);
	Texturemap::init(predict_rough_tex, predict_rast, predict_intr, predict_rough);
	Texturemap::init(predict_nor_tex, predict_rast, predict_intr, predict_nor);
	Texturemap::init(predict_disp_tex, predict_rast, predict_intr, predict_disp);
	Material::init(predict_mtr, predict_rast, predict_pos_proj, predict_normal_proj, &predict_texel, 3,
		predict_diff_tex.kernel.out, predict_rough_tex.kernel.out, predict_nor_tex.kernel.out, predict_disp_tex.kernel.out,
		predict_diff_tex.grad.out, predict_rough_tex.grad.out, predict_nor_tex.grad.out, predict_disp_tex.grad.out);
	PBRMaterial::init(predict_mtr, *(float3*)&mat.eye, 1, direction, lightintensity, 2.19f);
	Loss::init(loss, target_mtr.kernel.out, predict_mtr.kernel.out, predict_mtr.grad.out, width, height, 3);
	Optimizer::init(diff_adam, predict_diff);
	Optimizer::init(rough_adam, predict_rough);
	Optimizer::init(nor_adam, predict_nor);
	Adam::setHyperParams(diff_adam, 5e-3, .9, .999, 1e-8);
	Adam::setHyperParams(rough_adam, 5e-3, .9, .999, 1e-8);
	Adam::setHyperParams(nor_adam, 1e-2, .9, .999, 1e-8);

	GLbuffer::init(target_diff_buffer, target_diff.texture[0], target_diff.width, target_diff.height, 3, 15);
	GLbuffer::init(target_rough_buffer, target_rough.texture[0], target_rough.width, target_rough.height, 1, 14);
	GLbuffer::init(target_nor_buffer, target_nor.texture[0], target_nor.width, target_nor.height, 3, 13);
	GLbuffer::init(target_mtr_buffer, target_mtr.kernel.out, width, height, 3, 12);
	GLbuffer::init(predict_diff_buffer, predict_diff.texture[0], predict_diff.width, predict_diff.height, 3, 11);
	GLbuffer::init(predict_rough_buffer, predict_rough.texture[0], predict_rough.width, predict_rough.height, 1, 10);
	GLbuffer::init(predict_nor_buffer, predict_nor.texture[0], predict_nor.width, predict_nor.height, 3, 9);
	GLbuffer::init(predict_mtr_buffer, predict_mtr.kernel.out, width, height, 3, 8);
}

void PresetPrimitives::display(void) {
	Matrix::forward(mat);
	Project::forward(target_proj);
	Rasterize::forward(target_rast);
	Interpolate::forward(target_intr);
	Project::forward(target_pos_proj);
	Normalcalc::forward(target_norm);
	Project::forward(target_normal_proj);
	Texturemap::forward(target_diff_tex);
	Texturemap::forward(target_rough_tex);
	Texturemap::forward(target_nor_tex);
	Texturemap::forward(target_disp_tex);
	PBRMaterial::forward(target_mtr);

	Project::forward(predict_proj);
	Rasterize::forward(predict_rast);
	Interpolate::forward(predict_intr);
	Project::forward(predict_pos_proj);
	Normalcalc::forward(predict_norm);
	Project::forward(predict_normal_proj);
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
	TextureGrad::buildMIP(predict_diff);
	TextureGrad::clear(predict_diff);
	TextureGrad::gradSumup(predict_rough);
	Adam::step(rough_adam);
	TextureGrad::buildMIP(predict_rough);
	TextureGrad::clear(predict_rough);
	TextureGrad::gradSumup(predict_nor);
	Adam::step(nor_adam);
	TextureGrad::buildMIP(predict_nor);
	TextureGrad::clear(predict_nor);
	Texture::normalize(predict_nor);

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

void PresetPrimitives::update(double dt) {
	t += dt * .25;
	if (t > 6.2831853071795864)t -= 6.2831853071795864;
	Matrix::setRandomRotation(mat);
}