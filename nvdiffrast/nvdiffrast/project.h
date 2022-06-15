#pragma once
#include "common.h"
#include "buffer.h"

struct ProjectKernelParams {
	int vboNum;
	int dimention;
	float* vbo;
	float* mat;

	float* out;
};

struct ProjectKernelGradParams {
	float* out;
	float* mat;

	float* vbo;
};

struct ProjectParams{
	ProjectKernelParams kernel;
	unsigned int* vao;
	int vaoNum;
	size_t vboSize() { return (size_t)kernel.vboNum * kernel.dimention * sizeof(float); };
	size_t vaoSize() { return (size_t)vaoNum * 3 * sizeof(unsigned int); };
};

struct ProjectGradParams : ProjectParams {
	ProjectKernelGradParams grad;
};

class Project {
public:
	static void init(ProjectParams& proj, float* mat, Attribute& vec, bool homogeneous);
	static void init(ProjectParams& proj, float* mat, Attribute& vec, Attribute& out, bool homogeneous);
	static void init(ProjectGradParams& proj, float* mat, AttributeGrad& vec, bool homogeneous);
	static void init(ProjectGradParams& proj, float* mat, AttributeGrad& vec, AttributeGrad& out, bool homogeneous);
	static void forward(ProjectParams& proj);
	static void forward(ProjectGradParams& proj);
	static void backward(ProjectGradParams& proj);
};