#include "matrix.h"

void Rotation::init(RotationParams& rot) {
	rot.kernel.q = glm::quat(1.f, 0.f, 0.f, 0.f);
	CUDA_ERROR_CHECK(cudaMallocHost(&rot.kernel.rotation, 16 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&rot.kernel.out, 16 * sizeof(float)));
}

void Rotation::init(RotationGradParams& rot) {
	init((RotationParams&)rot);
	rot.grad.q = glm::quat(1.f, 0.f, 0.f, 0.f);
	CUDA_ERROR_CHECK(cudaMallocHost(&rot.grad.rotation, 16 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&rot.grad.out, 16 * sizeof(float)));
}

void Rotation::setRandom(RotationParams& rot) {
	float angle = ((float)rand() / (float)RAND_MAX * 2.f - 1.f) * 3.14159265f;
	float theta = ((float)rand() / (float)RAND_MAX * 2.f - 1.f) * 3.14159265f;
	float z = (float)rand() / (float)RAND_MAX * 2.f - 1.f;
	float c = sqrt(1.f - z * z);
	float x = c * cos(theta);
	float y = c * sin(theta);
	rot.kernel.q = glm::angleAxis(angle, glm::normalize(glm::vec3(x, y, z)));
}


void Rotation::setRotation(RotationParams& rot, float angle, float x, float y, float z){
	float s = sin(angle * .5f)/ sqrt(x * x + y * y + z * z);
	rot.kernel.q = glm::quat(cos(angle * .5f), s * x, s * y, s * z);
}

void Rotation::addRotation(RotationParams& rot, float angle, float x, float y, float z){
	float s = sin(angle * .5f)/ sqrt(x * x + y * y + z * z);
	rot.kernel.q = glm::quat(cos(angle * .5f), s * x, s * y, s * z) * rot.kernel.q;
}

//rot = 
// (qw*qw + qx*qx - qy*qy - qz*qz           2 * (qx*qy - qw*qz)           2 * (qz*qx + qw*qy))
// (          2 * (qx*qy + qw*qz) qw*qw - qx*qx + qy*qy - qz*qz           2 * (qy*qz - qw*qx))
// (          2 * (qz*qx - qw*qy)           2 * (qy*qz + qw*qx) qw*qw - qx*qx - qy*qy + qz*qz)
//

glm::mat4 getRotation(glm::quat q) {
	float qw2 = q.w * q.w, qx2 = q.x * q.x, qy2 = q.y * q.y, qz2 = q.z * q.z;
	float qwx = q.w * q.x, qwy = q.w * q.y, qwz = q.w * q.z;
	float qxy = q.x * q.y, qyz = q.y * q.z, qzx = q.z * q.x;
	return glm::mat4(
		qw2 + qx2 - qy2 - qz2, 2.f * (qxy + qwz), 2.f * (qzx - qwy), 0.f,
		2.f * (qxy - qwz), qw2 - qx2 + qy2 - qz2, 2.f * (qyz + qwx), 0.f,
		2.f * (qzx + qwy), 2.f * (qyz - qwx), qw2 - qx2 - qy2 + qz2, 0.f,
		0.f, 0.f, 0.f, 1.f
	);
}

void Rotation::forward(RotationParams& rot) {
	*rot.kernel.rotation = getRotation(rot.kernel.q);
	CUDA_ERROR_CHECK(cudaMemcpy(rot.kernel.out, rot.kernel.rotation, 16 * sizeof(float), cudaMemcpyHostToDevice));
}

void Rotation::forward(RotationGradParams& rot) {
	rot.grad.q = glm::quat();
	*rot.grad.rotation = glm::mat4();
	CUDA_ERROR_CHECK(cudaMemset(rot.grad.out, 0, 16 * sizeof(float)));
	forward((RotationParams&)rot);
}

// 
// dR/dqw = 2 * (qw -qz qy)
//              (qz qw -qx)
//              (-qy qx qw)
// 
// dR/dqx = 2 * (qx qy qz)
//              (qy -qx -qw)
//              (qz qw -qx)
// 
// dR/dqy = 2 * (-qy qx qw)
//              (qx qy qz)
//              (-qw qz -qy)
// 
// dR/dqz = 2 * (-qz -qw qx)
//              (qw -qz qy)
//              (qx qy qz)
// 

void Rotation::backward(RotationGradParams& rot) {
	if (*rot.grad.rotation == glm::mat4()) {
		CUDA_ERROR_CHECK(cudaMemcpy(rot.grad.rotation, rot.grad.out, 16 * sizeof(float), cudaMemcpyDeviceToHost));
	}
	glm::mat3 r = glm::mat3(*rot.grad.rotation);
	glm::quat q = rot.kernel.q;
	glm::vec3 wzy = glm::vec3(q.w, q.z, -q.y);
	glm::vec3 zwx = glm::vec3(-q.z, q.w, q.x);
	glm::vec3 yxw = glm::vec3(q.y, -q.x, q.w);
	glm::vec3 xyz = glm::vec3(q.x, q.y, q.z);
	rot.grad.q.w = glm::dot(r[0], wzy) + glm::dot(r[1], zwx) + glm::dot(r[2], yxw);
	rot.grad.q.x = glm::dot(r[0], xyz) + glm::dot(r[1], yxw) - glm::dot(r[2], zwx);
	rot.grad.q.y = -glm::dot(r[0], yxw) + glm::dot(r[1], xyz) + glm::dot(r[2], wzy);
	rot.grad.q.z = glm::dot(r[0], zwx) - glm::dot(r[1], wzy) + glm::dot(r[2], xyz);
}

void Rotation::init(RotationGradParams& rot, double eta, double rhom, double rhov, double eps) {
	rot.it = 0;
	rot.rhom = rhom;
	rot.rhov = rhov;
	rot.rhomt = rhom;
	rot.rhovt = rhov;
	rot.eta = eta;
	rot.eps = eps;
}

void Rotation::step(RotationGradParams& rot) {
	rot.it++;
	rot.rhomt *= rot.rhom;
	rot.rhovt *= rot.rhov;

	float d = sqrt(glm::dot(rot.grad.q, rot.grad.q));
	if (d > 0) {
		std::cout <<
			"distance :" << d <<
			" similar :" << glm::dot(rot.kernel.q, rot.grad.q) / d << std::endl;
	}
	else {
		std::cout <<
			"distance :" << d << std::endl;
	}


	glm::quat q = rot.kernel.q;
	for (int i = 0; i < 4; i++) {
		rot.m[i] = rot.rhom * rot.m[i] + (1 - rot.rhom) * rot.grad.q[i];
		rot.v[i] = rot.rhov * rot.v[i] + (1 - rot.rhov) * rot.grad.q[i] * rot.grad.q[i];
		double m = rot.m[i] / (1 - rot.rhomt);
		double v = rot.v[i] / (1 - rot.rhovt);
		q[i] -= m * rot.eta / (sqrt(v) + rot.eps);
	}
	rot.kernel.q = glm::normalize(q);
}

void Rotation::reset(RotationGradParams& rot){
	rot.rhomt = rot.rhom;
	rot.rhovt = rot.rhov;
	rot.m = glm::quat();
	rot.v = glm::quat();
}


void Camera::init(CameraParams& cam, RotationParams& rot, glm::vec3 eye, glm::vec3 center, glm::vec3 up, float size, float aspect, float znear, float zfar) {
	cam.kernel.size = size;
	cam.kernel.aspect = aspect;
	cam.kernel.znear = znear;
	cam.kernel.zfar = zfar;
	cam.kernel.rotation = rot.kernel.rotation;
	CUDA_ERROR_CHECK(cudaMallocHost(&cam.kernel.view, 16 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMallocHost(&cam.kernel.projection, 16 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMallocHost(&cam.kernel.mat, 16 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&cam.kernel.out, 16 * sizeof(float)));
	setCam(cam, eye, center, up);
}

void Camera::init(CameraGradParams& cam, RotationParams& rot, glm::vec3 eye, glm::vec3 direction, glm::vec3 up, float size, float aspect, float znear, float zfar) {
	init((CameraParams&)cam, rot, eye, direction, up, size, aspect, znear, zfar);
	cam.grad.eye = glm::vec3();
	cam.grad.size = 0.f;
	CUDA_ERROR_CHECK(cudaMallocHost(&cam.grad.view, 16 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMallocHost(&cam.grad.projection, 16 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMallocHost(&cam.grad.mat, 16 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&cam.grad.out, 16 * sizeof(float)));
}

void Camera::init(CameraGradParams& cam, RotationGradParams& rot, glm::vec3 eye, glm::vec3 direction, glm::vec3 up, float size, float aspect, float znear, float zfar) {
	init(cam, (RotationParams&)rot, eye, direction, up, size, aspect, znear, zfar);
	cam.grad.rotation = rot.grad.rotation;
}

glm::mat4 getView(glm::quat q, glm::vec3 eye) {
	glm::mat4 view = getRotation(q);
	view[3] = view * glm::vec4(-eye, 1.f);

	return view;
}

void Camera::setCam(CameraParams& cam, glm::vec3 eye, glm::vec3 center, glm::vec3 up) {
	cam.kernel.eye = eye;
	glm::vec3 f = glm::normalize(eye - center);
	glm::vec3 s = glm::normalize(glm::cross(up, f));
	glm::vec3 u = glm::normalize(glm::cross(f, s));

	cam.kernel.q.w = .5f * sqrt(s.x + u.y + f.z + 1.f);
	cam.kernel.q.x = (f.y > u.z ? .5f : -.5f) * sqrt(abs(s.x - u.y - f.z + 1.f));
	cam.kernel.q.y = (s.z > f.x ? .5f : -.5f) * sqrt(abs(-s.x + u.y - f.z + 1.f));
	cam.kernel.q.z = (u.x > s.y ? .5f : -.5f) * sqrt(abs(-s.x - u.y + f.z + 1.f));
}

//
// (near/size                  0          -ex/ez*near/size                                     0)
// (        0 near/(size*aspect) -ey/ez*near/(size*aspect)                                     0)
// (        0                  0    -(far+near)/(far-near) (ez*(far+near)-2*far*near)/(far-near))
// (        0                  0                        -1                                    ez)
// 
// dL/dex = -dL/dmat[2][0]/ez * near/size
// dL/dey = -dL/dmat[2][1]/ez * near/(size*aspect)
// dL/dez = (dL/dmat[2][0] * ex + dL/dmat[2][1] * ey/aspect)*near/size/ez^2 + dL/dmat[3][2] * (far+near)/(far-near) + dL/dmat[3][3]
// 
// dL/dsize = (-(dL/dmat[0][0] + dL/dmat[1][1]/aspect) + (dL/dmat[2][0] * ex + dL/dmat[2][1] * ey/aspect)/ez)*near/size^2
// dL/dnear = (dL/dmat[0][0] + dL/dmat[1][1]/aspect - (dL/dmat[2][0]*ex + dL/dmat[2][1]*ey/aspect)/ez)/size
//            + dL/dmat[2][2] * -2*far/(far-near)^2 + dL/dmat[3][2] * (2*far*(ez-far))/(far-near)^2
// dL/dfar = dL/dmat[2][2] * 2*near/(far-near)^2 - dL/dmat[3][2] * 2*near*(ez-near)/(far-near)^2
//
void Camera::forward(CameraParams& cam) {
	*cam.kernel.view = getView(cam.kernel.q, cam.kernel.eye);
	float sy = cam.kernel.znear / cam.kernel.size;
	float sx = sy / cam.kernel.aspect;
	float f_n = cam.kernel.zfar - cam.kernel.znear;
	*cam.kernel.projection = glm::mat4(
		sx, 0.f, 0.f, 0.f,
		0.f, sy, 0.f, 0.f,
		0.f, 0.f, -(cam.kernel.zfar + cam.kernel.znear) / f_n, -1.f,
		0.f, 0.f, -2.f * cam.kernel.zfar * cam.kernel.znear / f_n, 0.f
	);
	*cam.kernel.mat = *cam.kernel.projection * *cam.kernel.view * *cam.kernel.rotation;
	CUDA_ERROR_CHECK(cudaMemcpy(cam.kernel.out, cam.kernel.mat, 16 * sizeof(float), cudaMemcpyHostToDevice));
}

void Camera::forward(CameraGradParams& cam) {
	cam.grad.eye = glm::vec3();
	cam.grad.size = 0.f;
	CUDA_ERROR_CHECK(cudaMemset(cam.grad.view, 0, 16 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMemset(cam.grad.projection, 0, 16 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMemset(cam.grad.mat, 0, 16 * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMemset(cam.grad.out, 0, 16 * sizeof(float)));
	forward((CameraParams&)cam);
}

void Camera::backward(CameraGradParams& cam) {
	CUDA_ERROR_CHECK(cudaMemcpy(cam.grad.mat, cam.grad.out, 16 * sizeof(float), cudaMemcpyDeviceToHost));
	*cam.grad.rotation = *cam.grad.mat * glm::transpose(*cam.kernel.view) * glm::transpose(*cam.kernel.projection);
	*cam.grad.view = glm::transpose(*cam.kernel.rotation) * *cam.grad.mat * glm::transpose(*cam.kernel.projection);
	*cam.grad.projection = glm::transpose(*cam.kernel.rotation) * glm::transpose(*cam.kernel.view) * *cam.grad.mat;
}