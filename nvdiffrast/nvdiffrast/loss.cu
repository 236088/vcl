#include "loss.h"

void Loss::init(LossParams& loss, float* target, float* predict, float* grad, int width, int height, int depth) {
	loss.kernel.target = target;
	loss.kernel.predict = predict;
	loss.kernel.grad = grad;
	loss.kernel.width = width;
	loss.kernel.height = height;
	loss.kernel.depth = depth;
	loss.kernel.size = width * height * depth;
	CUDA_ERROR_CHECK(cudaMalloc(&loss.kernel.buffer, loss.Size()));
	loss.block = getBlock(width, height);
	loss.grid = getGrid(loss.block, width, height, depth);
	int msb = 0;
	if (loss.kernel.size & 0xffffffff00000000)msb += 32;
	if (loss.kernel.size & 0xffff0000ffff0000)msb += 16;
	if (loss.kernel.size & 0xff00ff00ff00ff00)msb += 8;
	if (loss.kernel.size & 0xf0f0f0f0f0f0f0f0)msb += 4;
	if (loss.kernel.size & 0xcccccccccccccccc)msb += 2;
	if (loss.kernel.size & 0xaaaaaaaaaaaaaaaa)msb += 1;
	int hmsb = msb / 2;
	--msb;
	loss.stride = 1 << msb;
	loss.lh = 1 << hmsb;
	loss.rh = 1 << (msb - hmsb);
}

__global__ void lossGradient(const LossKernelParams loss) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + loss.width * (py + loss.height * pz);
	if (pidx >= loss.size)return;
	loss.grad[pidx] = loss.predict[pidx] - loss.target[pidx];
	loss.buffer[pidx] = loss.grad[pidx]* loss.grad[pidx];
}

__global__ void reduction(const LossKernelParams loss, int width, int height, int stride) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + width * (py + height * pz);
	if (pidx + stride >= loss.size)return;
	loss.buffer[pidx] += loss.buffer[pidx + stride];
}

void MSELoss::backward(LossParams& loss) {
	void* args[] = { &loss.kernel };
	CUDA_ERROR_CHECK(cudaLaunchKernel(lossGradient, loss.grid, loss.block, args, 0, NULL));
	int stride = loss.stride, w = loss.lh, h = loss.rh;
	void* rargs[] = { &loss.kernel, &w, &h, &stride };
	while (stride > 0)
	{
		dim3 block = getBlock(w, h);
		dim3 grid = getGrid(block, w, h);
		CUDA_ERROR_CHECK(cudaLaunchKernel(reduction, grid, block, rargs, 0, NULL));
		stride >>= 1;
		if (h >= w)h >>= 1;
		else w >>= 1;
	}
	CUDA_ERROR_CHECK(cudaMemcpy(&loss.loss, loss.kernel.buffer, sizeof(float), cudaMemcpyDeviceToHost));
	loss.loss /= 2.f;
}