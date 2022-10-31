#pragma once
#include "common.h"

struct RotationKernelParams {
	glm::quat q;
	glm::mat4* rotation;

	float* out;
};

struct RotationKernelGradParams {
	glm::quat q;
	glm::mat4* rotation;

	float* out;
};

struct RotationParams {
	RotationKernelParams kernel;
};

struct RotationGradParams : RotationParams {
	int it;
	double rhom;
	double rhov;
	double rhomt;
	double rhovt;
	double eta;
	double eps;
	glm::quat m;
	glm::quat v;
	RotationKernelGradParams grad;
};

class Rotation {
public:
	static void init(RotationParams& rot);
	static void init(RotationGradParams& rot);
	static void setRandom(RotationParams& rot);
	static void setRotation(RotationParams& rot, float angle, float x, float y, float z);
	static void addRotation(RotationParams& rot, float angle, float x, float y, float z);
	static void forward(RotationParams& rot);
	static void forward(RotationGradParams& rot);
	static void backward(RotationGradParams& rot);
	static void init(RotationGradParams& rot, double eta, double rhom, double rhov, double eps);
	static void step(RotationGradParams& rot);
	static void reset(RotationGradParams& rot);
};




struct CameraKernelParams {
	glm::vec3 eye;
	glm::quat q;
	float size;
	float aspect;
	float znear;
	float zfar;
	glm::mat4* rotation;
	glm::mat4* view;
	glm::mat4* projection;
	glm::mat4* mat;

	float* out;
};

struct CameraKernelGradParams {
	glm::vec3 eye;
	glm::quat q;
	float size;
	glm::mat4* rotation;
	glm::mat4* view;
	glm::mat4* projection;
	glm::mat4* mat;

	float* out;
};

struct CameraParams {
	CameraKernelParams kernel;
};

struct CameraGradParams : CameraParams {
	CameraKernelGradParams grad;
};

class Camera {
public:
	static void init(CameraParams& cam, RotationParams& rot, glm::vec3 eye, glm::vec3 direction, glm::vec3 up, float size, float aspect, float znear, float zfar);
	static void init(CameraGradParams& cam, RotationParams& rot, glm::vec3 eye, glm::vec3 direction, glm::vec3 up, float size, float aspect, float znear, float zfar);
	static void init(CameraGradParams& cam, RotationGradParams& rot, glm::vec3 eye, glm::vec3 direction, glm::vec3 up, float size, float aspect, float znear, float zfar);
	static void setCam(CameraParams& cam, glm::vec3 eye, glm::vec3 direction, glm::vec3 up);
	static void forward(CameraParams& cam);
	static void forward(CameraGradParams& cam);
	static void backward(CameraGradParams& cam);
};