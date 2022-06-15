#pragma once
#include "common.h"

struct TransformParams {
	glm::quat quaternion;
	glm::mat4 rotation;
	float* out;
};

struct TransformGradParams : TransformParams{
	float* out;
	glm::mat4 rotation;
	glm::quat quaternion;
};

class Transform {
public:
	static void init(TransformParams& tf, float angle, float x, float y, float z);
	static void init(TransformGradParams& tf, float angle, float x, float y, float z);
	static void forward(TransformParams& tf);
	static void forward(TransformGradParams& tf);
	static void backward(TransformGradParams& tf);
	static void setRotation(TransformParams& tf, float angle, float x, float y, float z);
	static void lookAt(TransformParams& tf, float x, float y, float z);
	static void addRotation(TransformParams& tf, float angle, float x, float y, float z);
	static void setRandomRotation(TransformParams& tf);
};



struct CameraParams {
	glm::vec3 eye;
	float centerx;
	float centery;
	float size;
	float aspect;
	float znear;
	float zfar;
	float* rot;
	glm::mat4 mat;
	float* out;
};

struct CameraGradParams :CameraParams {
	glm::vec3 eye;
	float centerx;
	float centery;
	float size;
	float* rot;
	glm::mat4 mat;
	float* out;
};

class Camera {
public:
	static void init(CameraParams& cam, TransformParams& tf, float ex, float ey, float ez, float centerx, float centery, float size, float aspect, float znear, float zfar);
	static void init(CameraGradParams& cam, TransformParams& tf, float ex, float ey, float ez, float centerx, float centery, float size, float aspect, float znear, float zfar);
	static void init(CameraGradParams& cam, TransformGradParams& tf, float ex, float ey, float ez, float centerx, float centery, float size, float aspect, float znear, float zfar);
	static void forward(CameraParams& cam);
	static void forward(CameraGradParams& cam);
	static void backward(CameraGradParams& cam);
	static void setEye(CameraParams& cam, float ex, float ey, float ez);
	static void setProjection(CameraParams& cam, float centerx, float centery, float size, float aspect, float znear, float zfar);
};