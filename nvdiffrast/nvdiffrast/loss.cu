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

__device__ __forceinline__ void calculateRange(int p, int k, int w, int& s, int& e) {
	s = p - k;
	if (s < 0) s = 0;
	e = p + k;
	if (e >= w) e = w - 1;
}

__global__ void SSIMlossGradient(const LossKernelParams loss, int k, float c1, float c2, float invk2) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= loss.width || py >= loss.height || pz >= loss.depth)return;
	int pidx = (px + loss.width * py) * loss.depth + pz;
	int sx, ex, sy, ey;
	calculateRange(px, k, loss.width, sx, ex);
	calculateRange(py, k, loss.height, sy, ey);

	float predictAve = 0.f;
	float targetAve = 0.f;
	for (int ix = sx; ix <= ex; ix++) {
		for (int iy = sy; iy <= ey; iy++) {
			int idx = ix + loss.width * iy;
			predictAve += loss.predict[idx * loss.depth + pz];
			targetAve += loss.target[idx * loss.depth + pz];
		}
	}
	predictAve *= invk2;
	targetAve *= invk2;

	float predictVar = 0.f;
	float targetVar = 0.f;
	float Cov = 0.f;

	for (int ix = sx; ix <= ex; ix++) {
		for (int iy = sy; iy <= ey; iy++) {
			int idx = ix + loss.width * iy;
			float predictDiff = loss.predict[idx * loss.depth + pz] - predictAve;
			float targetDiff = loss.target[idx * loss.depth + pz] - targetAve;
			predictVar += predictDiff * predictDiff;
			targetVar += targetDiff * targetDiff;
			Cov += predictDiff * targetDiff;
		}
	}

	predictVar *= invk2;
	targetVar *= invk2;
	Cov *= invk2;

	float numer1 = 2.f * predictAve * targetAve + c1;
	float denom1 = predictAve * predictAve + targetAve * targetAve + c1;
	float numer2 = 2.f * Cov + c2;
	float denom2 = predictVar + targetVar + c2;
	float denomInv = 1.f / (denom1 * denom2);
	loss.buffer[pidx] = 1.f - numer1 * numer2 * denomInv;
	loss.grad[pidx] = -(targetAve * numer2 + numer1 * (loss.target[pidx] - targetAve)
		- loss.buffer[pidx] * (predictAve * denom2 + denom1 * (loss.predict[pidx] - predictAve)))
		* 2.f * invk2 * denomInv;
}

void SSIMLoss::backward(LossParams& loss) {
	float c1 = 1e-4, c2 = 1e-3;
	int k = 5;
	float invk2 = 1.f / (float)(k * k);
	void* args[] = { &loss.kernel, &k, &c1, &c2, &invk2 };
	CUDA_ERROR_CHECK(cudaLaunchKernel(SSIMlossGradient, loss.grid, loss.block, args, 0, NULL));
	int msb = MSB(loss.kernel.size);
	int hmsb = msb / 2;
	--msb;
	int stride = 1 << msb, w = 1 << hmsb, h = 1 << (msb - hmsb);
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
	loss.loss /= loss.Size();
}