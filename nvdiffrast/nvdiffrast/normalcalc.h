#pragma once
#include "common.h"
#include "buffer.h"

struct NormalcalcKernelParams {
	int posNum;
	float* pos;
	unsigned int* vao;
	int vaoNum;
	float* mat;

	float* out;
};

struct NormalcalcKernelGradParams {
	float* out;

	float* pos;
};

struct NormalcalcParams {
	NormalcalcKernelParams kernel;
	dim3 vaoblock;
	dim3 vaogrid;
	dim3 block;
	dim3 grid;
	size_t posSize() { return (size_t)kernel.posNum * 3 * sizeof(float); };
	size_t vaoSize() { return (size_t)kernel.vaoNum * 3 * sizeof(unsigned int); };
};

struct NormalcalcGradParams : NormalcalcParams {
	NormalcalcKernelGradParams grad;
};

class Normalcalc {
public:
	static void init(NormalcalcParams& norm, Attribute& pos, Attribute& normal);
	static void init(NormalcalcGradParams& norm, AttributeGrad& pos, AttributeGrad& normal);
	static void forward(NormalcalcParams& norm);
	static void forward(NormalcalcGradParams& norm);
	static void backward(NormalcalcGradParams& norm);
};
