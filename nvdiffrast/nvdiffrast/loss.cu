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
	int msb = MSB(loss.kernel.size);
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

__global__ void texturelosskernel(const LossKernelParams loss) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = (px + loss.width * py) * loss.depth + pz;
	if (pidx >= loss.size)return;
	int w = loss.width / 4;
	int w2 = loss.width / 2;
	if ((px < w || w2 <= px) && (py < w || w2 <= py)) {
		loss.buffer[pidx] = 0.f;
		return;
	}
	float t = loss.predict[pidx] - loss.target[pidx];
	loss.buffer[pidx] = t * t;
}

void MSELoss::textureloss(LossParams& loss) {
	void* args[] = { &loss.kernel };
	CUDA_ERROR_CHECK(cudaLaunchKernel(texturelosskernel, loss.grid, loss.block, args, 0, NULL));
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
	loss.loss /= (loss.kernel.size / 2);
}