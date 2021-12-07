#pragma once
#include "common.h"
#include "buffer.h"

struct LossKernelParams {
	float* predict;
	float* target;
	float* grad;
	float* buffer;

	int width;
	int height;
	int depth;
	int size;
};

struct LossParams {
	LossKernelParams kernel;
	float loss;
	int stride;
	int lh, rh;
	dim3 block;
	dim3 grid;
	size_t Size() { return (size_t)kernel.width * kernel.height * kernel.depth * sizeof(float); };
};

class Loss {
public:
	static void init(LossParams& loss, float* target, float* predict, float* grad, int width, int height, int depth);
	static float loss(LossParams& loss) { return loss.loss; };
};

class MSELoss : Loss{
public:
	static void backward(LossParams& loss);
};