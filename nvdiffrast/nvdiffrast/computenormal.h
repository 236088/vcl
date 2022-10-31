#pragma once
#include "common.h"
#include "buffer.h"

struct ComputeNormalKernelParams {
	int posNum;
	float* pos;
	unsigned int* vao;
	int vaoNum;
	float* mat;

	float* out;
};

struct ComputeNormalKernelGradParams {
	float* pos;

	float* out;
};

struct ComputeNormalParams {
	ComputeNormalKernelParams kernel;
	dim3 vaoblock;
	dim3 vaogrid;
	dim3 block;
	dim3 grid;
	size_t posSize() { return (size_t)kernel.posNum * 3 * sizeof(float); };
	size_t vaoSize() { return (size_t)kernel.vaoNum * 3 * sizeof(unsigned int); };
};

struct ComputeNormalGradParams : ComputeNormalParams {
	ComputeNormalKernelGradParams grad;
};

class ComputeNormal {
public:
	static void init(ComputeNormalParams& norm, Attribute& pos, Attribute& normal);
	static void init(ComputeNormalGradParams& norm, AttributeGrad& pos, AttributeGrad& normal);
	static void forward(ComputeNormalParams& norm);
	static void forward(ComputeNormalGradParams& norm);
	static void backward(ComputeNormalGradParams& norm);
};
