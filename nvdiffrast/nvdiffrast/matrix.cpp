#include "matrix.h"

void Transform::init(TransformParams& tf, float angle, float x, float y, float z) {
	setRotation(tf, angle, x, y, z);
	CUDA_ERROR_CHECK(cudaMalloc(&tf.out, 16 * sizeof(float)));
}
void Transform::init(TransformGradParams& tf, float angle, float x, float y, float z) {
	init((TransformParams&)tf, angle, x, y, z);
	tf.quaternion = glm::quat();
	CUDA_ERROR_CHECK(cudaMalloc(&tf.out, 16 * sizeof(float)));
}

void Transform::forward(TransformParams& tf) {
	tf.rotation = glm::toMat4(tf.quaternion);
	CUDA_ERROR_CHECK(cudaMemcpy(tf.out, &tf.rotation, 16 * sizeof(float), cudaMemcpyHostToDevice));
}

void Transform::forward(TransformGradParams& tf) {
	tf.quaternion = glm::quat();
	tf.rotation = glm::mat4();
	CUDA_ERROR_CHECK(cudaMemset(tf.out, 0.f, 16 * sizeof(float)));
	forward((TransformParams&)tf);
}

void Transform::backward(TransformGradParams& tf) {
	CUDA_ERROR_CHECK(cudaMemcpy(&tf.rotation, tf.out, 16 * sizeof(float), cudaMemcpyDeviceToHost));
	glm::mat3 r = glm::mat3(tf.rotation);
	glm::quat q = ((TransformParams)tf).quaternion * 2.f;
	glm::vec3 xyz = glm::vec3(q.x, q.y, q.z);
	glm::vec3 yxw = glm::vec3(q.y, -q.x, q.w);
	glm::vec3 wzy = glm::vec3(q.w, q.z, -q.y);
	glm::vec3 zwx = glm::vec3(-q.z, q.w, q.x);
	tf.quaternion.x = glm::dot(r[0], xyz) + glm::dot(r[1], yxw) - glm::dot(r[2], zwx);
	tf.quaternion.y = -glm::dot(r[0], yxw) + glm::dot(r[1], xyz) + glm::dot(r[2], wzy);
	tf.quaternion.z = glm::dot(r[0], zwx) - glm::dot(r[1], wzy) + glm::dot(r[2], xyz);
	tf.quaternion.w = glm::dot(r[0], wzy) + glm::dot(r[1], zwx) + glm::dot(r[2], yxw);
}

void Transform::setRotation(TransformParams& tf, float angle, float x, float y, float z) {
	tf.quaternion = glm::angleAxis(angle, glm::normalize(glm::vec3(x, y, z)));
}

void Transform::lookAt(TransformParams& tf, float x, float y, float z) {
	tf.quaternion = glm::quatLookAt(-glm::normalize(glm::vec3(x, y, z)), glm::vec3(0.f, 1.f, 0.f));
}

void Transform::addRotation(TransformParams& tf, float angle, float x, float y, float z) {
	tf.quaternion = glm::rotate(tf.quaternion, angle, glm::vec3(x, y, z));
}

void Transform::setRandomRotation(TransformParams& tf) {
	float angle = ((float)rand() / (float)RAND_MAX * 2.f - 1.f) * 3.14159265f;
	float theta = ((float)rand() / (float)RAND_MAX * 2.f - 1.f) * 3.14159265f;
	float z = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
	float c = sqrt(1.f - z * z);
	float x = c * cos(theta);
	float y = c * sin(theta);
	setRotation(tf, angle, x, y, z);
}



void Camera::init(CameraParams& cam, TransformParams& tf, float ex, float ey, float ez, float centerx, float centery, float size, float aspect, float znear, float zfar) {
	cam.rot = tf.out;
	setEye(cam, ex, ey, ez);
	setProjection(cam, centerx, centery, size, aspect, znear, zfar);
	CUDA_ERROR_CHECK(cudaMalloc(&cam.out, 16 * sizeof(float)));
}

void Camera::init(CameraGradParams& cam, TransformParams& tf, float ex, float ey, float ez, float centerx, float centery, float size, float aspect, float znear, float zfar) {
	init((CameraParams&)cam, tf, ex, ey, ez, centerx, centery, size, aspect, znear, zfar);
	CUDA_ERROR_CHECK(cudaMalloc(&cam.out, 16 * sizeof(float)));
}

void Camera::init(CameraGradParams& cam, TransformGradParams& tf, float ex, float ey, float ez, float centerx, float centery, float size, float aspect, float znear, float zfar) {
	init(cam, (TransformParams&)tf, ex, ey, ez, centerx, centery, size, aspect, znear, zfar);
	CUDA_ERROR_CHECK(cudaMalloc(&cam.out, 16 * sizeof(float)));
	tf.out = cam.rot;
}

void Camera::forward(CameraParams& cam) {
	float left = cam.centerx - cam.size * cam.aspect;
	float right = cam.centerx + cam.size * cam.aspect;
	float bottom = cam.centery - cam.size;
	float top = cam.centery + cam.size;
	glm::mat4 rot;
	CUDA_ERROR_CHECK(cudaMemcpy(&rot, cam.rot, 16 * sizeof(float), cudaMemcpyDeviceToHost));
	glm::mat4 frustum = glm::mat4(
		cam.znear / (cam.size * cam.aspect), 0.f, 0.f, 0.f,
		0.f, cam.znear / cam.size, 0.f, 0.f,
		cam.centerx / (cam.size * cam.aspect), cam.centery / cam.size, -(cam.zfar + cam.znear) / (cam.zfar - cam.znear), -1.f,
		0.f, 0.f, -(2.f * cam.zfar * cam.znear) / (cam.zfar - cam.znear), 0.f
	);
	cam.mat = frustum * glm::lookAt(cam.eye, glm::vec3(cam.eye.x, cam.eye.y, 0.f), glm::vec3(0.f, 1.f, 0.f)) * rot;
	CUDA_ERROR_CHECK(cudaMemcpy(cam.out, &cam.mat, 16 * sizeof(float), cudaMemcpyHostToDevice));
}

void Camera::forward(CameraGradParams& cam) {
	cam.eye = glm::vec3();
	cam.centerx = 0.f;
	cam.centery = 0.f;
	cam.size = 0.f;
	cam.mat = glm::mat4();
	CUDA_ERROR_CHECK(cudaMemset(cam.out, 0.f, 16 * sizeof(float)));
	forward((CameraParams&)cam);
}

void Camera::backward(CameraGradParams& cam) {

}

void Camera::setEye(CameraParams& cam, float ex, float ey, float ez) {
	cam.eye = glm::vec3(ex, ey, ez);
}

void Camera::setProjection(CameraParams& cam, float centerx, float centery, float size, float aspect, float znear, float zfar) {
	cam.centerx = centerx;
	cam.centery = centery;
	cam.size = size;
	cam.aspect = aspect;
	cam.znear = znear;
	cam.zfar = zfar;
}