#include "preset.h"

void PresetPrimitives::init() {
	int resolution =512;
	//Attribute::loadOBJ("../../models/monkey.obj", &_pos, &_texel, &_normal);
	//Attribute::loadOBJ("../../models/sphere.obj", &_pos, &_texel, &_normal);
	Attribute::loadOBJ("../../models/Infinite.obj", &_pos, &_texel, &_normal);
	//Attribute::loadOBJ("../../models/food_apple_01_1k.obj", &_pos, &_texel, &_normal);
	Attribute::init(color, _pos, 3);
	Attribute::addRandom(color, 0.f, 1.f);
	Attribute::step(color, .5f);
	Texture::loadBMP("../../textures/Infinite_Color.bmp", _diffusemap, 8);
	Texture::loadBMP("../../textures/Infinite_NormalGL.bmp", _normalmap, 8);
	Texture::loadBMP("../../textures/Infinite_Roughness.bmp", _roughnessmap, 8);
	//Texture::loadBMP("../../textures/food_apple_01_diff_1k.bmp", _diffusemap, 8);
	//Texture::loadBMP("../../textures/food_apple_01_nor_gl_1k.bmp", _normalmap, 8);
	//Texture::loadBMP("../../textures/food_apple_01_rough_1k.bmp", _roughnessmap, 8);
	//Texture::loadBMP("../../textures/rocks_ground_01_Color.bmp", _diffusemap, 8);
	//Texture::loadBMP("../../textures/rocks_ground_01_NormalGL.bmp", _normalmap, 8);
	//Texture::loadBMP("../../textures/rocks_ground_01_Roughness.bmp", _roughnessmap, 8);
	Texture::liner(_normalmap, 2.f, -1.f);
	Texture::init(sgbake, 1024, 512, 3, 1);

	SGBuffer::loadTXT("../../data/envmap2.txt", &sgbuf);
	//SGBuffer::randomize(sgbuf);

	SGBuffer::bake(sgbuf, sgbake);
	GLbuffer::init(bake_buffer, sgbake.texture[0], 1024, 512, 3);

	Rotation::init(rot);
	Rotation::setRotation(rot, -3.14159265f/6.f, 0.f, 1.f, 0.f);
	Camera::init(cam, rot, glm::vec3(0.f, 0.f, -.35f), glm::vec3(0.f, 0.f, 1.f), glm::vec3(0.f, 1.f, 0.f), .05f, 1.f, .1f, 10.f);

	Project::init(proj, cam.kernel.out, _pos, true);
	Rasterize::init(rast, proj, resolution, resolution, 1, true);
	Interpolate::init(intr, rast, _texel);
	Interpolate::init(color_intr, rast, color);
	Texturemap::init(diffusemap, rast, intr, _diffusemap);
	Texturemap::init(normalmap, rast, intr, _normalmap);
	NormalAxis::init(normal_axis, rot, rast, _normal, _pos, _texel, normalmap);
	ViewAxis::init(view_axis, rot, cam, rast);
	Texturemap::init(roughnessmap, rast, intr, _roughnessmap);
	SphericalGaussian::init(sg ,rast, normal_axis, view_axis, diffusemap, roughnessmap, sgbuf, 1.57f);
	Antialias::init(aa, rast, proj, sg.kernel.out, 3);
	Filter::init(flt, rast, aa.kernel.out, 3, 16);
	Rasterize::wireframeinit(wire, proj, resolution, resolution);


	GLbuffer::init(rast_buffer, rast.kernel.out, resolution, resolution, 4);
	GLbuffer::init(intr_buffer, intr.kernel.out, resolution, resolution, 2);
	GLbuffer::init(color_buffer, color_intr.kernel.out, resolution, resolution, 3);
	GLbuffer::init(tex_buffer, roughnessmap.kernel.out, resolution, resolution, 1);
	GLbuffer::init(normal_buffer, normal_axis.kernel.out, resolution, resolution, 3);
	GLbuffer::init(sgdiffenv_buffer, sg.kernel.outDiffenv, resolution, resolution, 3);
	GLbuffer::init(sgspecenv_buffer, sg.kernel.outSpecenv, resolution, resolution, 3);
	GLbuffer::init(sg_buffer, diffusemap.kernel.out, resolution, resolution, 3);
	GLbuffer::init(aa_buffer, aa.kernel.out, resolution, resolution, 3);
	GLbuffer::init(flt_buffer, flt.kernel.out, resolution, resolution, 3);
	GLbuffer::init(wire_buffer, wire.kernel.out, resolution, resolution, 4);

}

void PresetPrimitives::display(void) {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(normal_buffer, GL_RGB32F, GL_RGB, -1.f, 0.f, -.5f, 1.f);
	GLbuffer::draw(tex_buffer, GL_LUMINANCE, GL_RED, -.5f, 0.f, -0.f, 1.f);
	GLbuffer::draw(sgdiffenv_buffer, GL_RGB32F, GL_RGB, 0.f, 0.f, .5f, 1.f);
	GLbuffer::draw(sgspecenv_buffer, GL_RGB32F, GL_RGB, .5f, 0.f, 1.f, 1.f);
	GLbuffer::draw(bake_buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, -0.f, 0.f);
	GLbuffer::draw(wire_buffer, GL_RGBA32F, GL_RGBA, 0.f, -1.f, .5f, 0.f);
	GLbuffer::draw(aa_buffer, GL_RGB32F, GL_RGB, .5f, -1.f, 1.f, 0.f);
	glFlush();
}

void PresetPrimitives::update(double dt, double t, unsigned int step, bool& play) {
	//Rotation::addRotation(rot, dt * .1f, cos(t), .1f, sin(t));

	Rotation::forward(rot);
	Camera::forward(cam);

	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Interpolate::forward(color_intr);
	Texturemap::forward(diffusemap);
	Texturemap::forward(normalmap);
	NormalAxis::forward(normal_axis);
	ViewAxis::forward(view_axis);
	Texturemap::forward(roughnessmap);
	SphericalGaussian::forward(sg);
	Antialias::forward(aa);
	Filter::forward(flt);
	Rasterize::drawforward(wire);
}