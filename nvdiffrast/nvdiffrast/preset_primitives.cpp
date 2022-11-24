#include "preset.h"

#define LEARN_TEXTURE
#define LEARN_SG

void PresetPrimitives::init() {
	int resolution =512;
	//Attribute::loadOBJ("../../models/monkey.obj", &_pos, &_texel, &_normal);
	//Attribute::loadOBJ("../../models/sphere.obj", &_pos, &_texel, &_normal);
	Attribute::loadOBJ("../../models/Infinite.obj", &_pos, &_texel, &_normal);
	//Attribute::loadOBJ("../../models/food_apple_01_1k.obj", &_pos, &_texel, &_normal);
	Attribute::init(color, _pos, 3);
	Attribute::addRandom(color, 0.f, 1.f);
	Attribute::step(color, .5f);
	Texture::loadBMP("../../textures/Infinite_Color.bmp", target_diffusemap, 3);
	Texture::loadBMP("../../textures/Infinite_NormalGL.bmp", _normalmap, 3);
	Texture::loadBMP("../../textures/Infinite_Roughness.bmp", _roughnessmap, 3);
	//Texture::loadBMP("../../textures/food_apple_01_diff_1k.bmp", _diffusemap, 3);
	//Texture::loadBMP("../../textures/food_apple_01_nor_gl_1k.bmp", _normalmap, 3);
	//Texture::loadBMP("../../textures/food_apple_01_rough_1k.bmp", _roughnessmap, 3);
	//Texture::loadBMP("../../textures/rocks_ground_01_Color.bmp", _diffusemap, 3);
	//Texture::loadBMP("../../textures/rocks_ground_01_NormalGL.bmp", _normalmap, 3);
	//Texture::loadBMP("../../textures/rocks_ground_01_Roughness.bmp", _roughnessmap, 3);
	//Texture::loadBMP("../../textures/Tiles074_1K_Color.bmp", _diffusemap, 3);
	//Texture::loadBMP("../../textures/Tiles074_1K_NormalGL.bmp", _normalmap, 3);
	//Texture::loadBMP("../../textures/Tiles074_1K_Roughness.bmp", _roughnessmap, 3);
	Texture::liner(_normalmap, 2.f, -1.f);
	GLbuffer::init(target_map_buffer, target_diffusemap.texture[0], target_diffusemap.width, target_diffusemap.height, target_diffusemap.channel);
#ifdef LEARN_TEXTURE	
	TextureGrad::init(predict_diffusemap, 1024, 1024, 3, 3);
	float col[3] = { .75f, .60f, .55f };
	Texture::setColor(predict_diffusemap, col);
	GLbuffer::init(predict_map_buffer, predict_diffusemap.texture[0], 1024, 1024, 3);
#endif // LEARN_TEXTURE

	SGBuffer::loadTXT("../../data/envmap2.txt", &target_sgbuf);
	SGBufferGrad::init(predict_sgbuf, 16, 3);
	SGBuffer::randomize(predict_sgbuf);

	Texture::init(target_sgbake, 1024, 512, 3, 1);
	SGBuffer::bake(target_sgbuf, target_sgbake);
	GLbuffer::init(target_bake_buffer, target_sgbake.texture[0], 1024, 512, 3);

	Texture::init(predict_sgbake, 1024, 512, 3, 1);
	SGBuffer::bake(predict_sgbuf, predict_sgbake);
	GLbuffer::init(predict_bake_buffer, predict_sgbake.texture[0], 1024, 512, 3);

	Rotation::init(rot);
	Rotation::setRotation(rot, 1.5707963f, 0.f, 1.f, 0.f);
	Camera::init(cam, rot, glm::vec3(0.f, 0.f, -.35f), glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f), .05f, 1.f, .1f, 10.f);

	Project::init(proj, cam.kernel.out, _pos, true);
	Rasterize::init(rast, proj, resolution, resolution, 1, true);
	Interpolate::init(intr, rast, _texel);
	Interpolate::init(color_intr, rast, color);
	Texturemap::init(target_diff, rast, intr, target_diffusemap);
#ifdef LEARN_TEXTURE
	Texturemap::init(predict_diff, rast, intr, predict_diffusemap);
#endif
	Texturemap::init(normalmap, rast, intr, _normalmap);
	Texturemap::init(roughnessmap, rast, intr, _roughnessmap);
	NormalAxis::init(normal_axis, rot, rast, _normal, _pos, _texel, normalmap);
	ViewAxis::init(view_axis, rot, cam, rast);
	SGSpecular::init(spec, rast, normal_axis, view_axis, roughnessmap, 1.45f);
	SphericalGaussian::init(target_sg ,rast, normal_axis, view_axis, target_diff, spec, target_sgbuf);
	Antialias::init(target_aa, rast, proj, target_sg.kernel.out, 3);
#ifdef LEARN_TEXTURE
	SphericalGaussian::init(predict_sg ,rast, normal_axis, view_axis, predict_diff, spec, predict_sgbuf);
#else
	SphericalGaussian::init(predict_sg, rast, normal_axis, view_axis, target_diff, spec, predict_sgbuf);
#endif // LEARN_TEXTURE
	Antialias::init(predict_aa, rast, proj, predict_sg.kernel.out, predict_sg.grad.out, 3);
	Loss::init(loss, target_aa.kernel.out, predict_aa.kernel.out, predict_aa.grad.out, resolution, resolution, 3);

	Optimizer::init(adam_amplitude, ((SGBuffer&)predict_sgbuf).amplitude, predict_sgbuf.amplitude, predict_sgbuf.num * predict_sgbuf.channel, predict_sgbuf.num, predict_sgbuf.channel, 1);
	Optimizer::init(adam_sharpness, ((SGBuffer&)predict_sgbuf).sharpness, predict_sgbuf.sharpness, predict_sgbuf.num, predict_sgbuf.num, 1, 1);
	Optimizer::init(adam_axis, ((SGBuffer&)predict_sgbuf).axis, predict_sgbuf.axis, predict_sgbuf.num * 3, predict_sgbuf.num, 3, 1);

	Adam::setHyperParams(adam_amplitude, 1e-2, .9, .999, 1e-8);
	Adam::setHyperParams(adam_sharpness, 1e-2, .9, .999, 1e-8);
	Adam::setHyperParams(adam_axis, 1e-2, .9, .999, 1e-8);
#ifdef LEARN_TEXTURE
	Optimizer::init(adam_diffusemap, predict_diffusemap);
	Adam::setHyperParams(adam_diffusemap, 1e-3, .9, .999, 1e-8);
#endif

	GLbuffer::init(target_aa_buffer, target_aa.kernel.out, resolution, resolution, 3);
	GLbuffer::init(predict_aa_buffer, predict_aa.kernel.out, resolution, resolution, 3);

	Mat img;
	CUDA_ERROR_CHECK(cudaMallocHost(&buf, resolution * resolution * 3 * sizeof(float)));
	out = Mat(resolution, resolution, CV_32FC3, buf);
	img = imread("../../img/Lenna.jpg");
	img.convertTo(img, CV_32FC3, 1.0 / 255.0);
	cvtColor(img, img, COLOR_BGR2RGB);
	flip(img, img, 0);
	resize(img, out, cv::Size(), resolution / (double)img.cols, resolution / (double)img.rows);
	GLbuffer::init(sample_buffer, buf, resolution, resolution, 3);

	writer.open("C:/Users/s2360/Videos/Sample.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, Size(resolution, resolution));
	frame = Mat(resolution, resolution, CV_32FC3, predict_aa_buffer.gl_buffer);
	frm = Mat(resolution, resolution, CV_8UC3);
}

void PresetPrimitives::display(void) {

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(0);

	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_TEXTURE_2D);
	GLbuffer::draw(target_bake_buffer, GL_RGB32F, GL_RGB, -1.f, -1.f, 0.f, 0.f);
	GLbuffer::draw(target_map_buffer, GL_RGB32F, GL_RGB, 0.f, -1.f, .5f, 0.f);
	GLbuffer::draw(target_aa_buffer, GL_RGB32F, GL_RGB, .5f, -1.f, 1.f, 0.f);
	GLbuffer::draw(predict_bake_buffer, GL_RGB32F, GL_RGB, -1.f, 0.f, 0.f, 1.f);
#ifdef LEARN_TEXTURE
	GLbuffer::draw(predict_map_buffer, GL_RGB32F, GL_RGB, 0.f, 0.f, .5f, 1.f);
#endif
	GLbuffer::draw(predict_aa_buffer, GL_RGB32F, GL_RGB, .5f, 0.f, 1.f, 1.f);
	glFlush();
}

void PresetPrimitives::update(double dt, double t, unsigned int step, bool& play) {
	glm::vec3 e = .35f * glm::vec3(sin(t * .3f), 0.f, cos(t * .3f));
	Camera::setCam(cam, e, glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f));

	Rotation::forward(rot);
	Camera::forward(cam);

	Project::forward(proj);
	Rasterize::forward(rast);
	Interpolate::forward(intr);
	Interpolate::forward(color_intr);
	Texturemap::forward(target_diff);
#ifdef LEARN_TEXTURE
	Texturemap::forward(predict_diff);
#endif // LEARN_TEXTURE
	Texturemap::forward(normalmap);
	Texturemap::forward(roughnessmap);
	NormalAxis::forward(normal_axis);
	ViewAxis::forward(view_axis);
	SGSpecular::forward(spec);
	SphericalGaussian::forward(target_sg);
	Antialias::forward(target_aa);
	SphericalGaussian::forward(predict_sg);
	Antialias::forward(predict_aa);
	MSELoss::backward(loss);
	Antialias::backward(predict_aa);
	SphericalGaussian::backward(predict_sg);
#ifdef LEARN_TEXTURE
	Texturemap::backward(predict_diff);

	Adam::step(adam_diffusemap);
	TextureGrad::clear(predict_diffusemap);
#endif // LEARN_TEXTURE

	Adam::step(adam_amplitude);
	SGBufferGrad::clear(predict_sgbuf);
	SGBuffer::normalize(predict_sgbuf);

	Adam::step(adam_sharpness);
	Adam::step(adam_axis);


	SGBuffer::bake(predict_sgbuf, predict_sgbake);

	cout << "loss:" << Loss::loss(loss);

	frame.convertTo(frm, CV_8UC3, 255.0);
	cvtColor(frm, frm, COLOR_RGB2BGR);
	flip(frm, frm, 0);
	imwrite("C:/Users/s2360/Videos/Sample.png", frm);
	writer << frm;
}