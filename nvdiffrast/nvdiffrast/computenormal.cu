#include "computenormal.h"

void ComputeNormal::init(ComputeNormalParams& norm, Attribute& pos, Attribute& normal) {
	if (pos.dimention != 3)ERROR_STRING(dimention is not 3);
	Attribute::init(normal, pos, 3);
	norm.kernel.posNum = pos.vboNum;
	norm.kernel.pos = pos.vbo;
	norm.kernel.vao = pos.vao;
	norm.kernel.vaoNum = pos.vaoNum;
	norm.kernel.out = normal.vbo;
	norm.vaoblock = getBlock(pos.vaoNum, 1);
	norm.vaogrid = getGrid(norm.vaoblock, pos.vaoNum, 1);
	norm.block = getBlock(pos.vboNum, 1);
	norm.grid = getGrid(norm.block, pos.vboNum, 1);
}

__global__ void ComputeNormalForwardKernel(const ComputeNormalKernelParams norm) {
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx >= norm.vaoNum)return;
	uint3 tri = ((uint3*)norm.vao)[pidx];
	float3 v0 = ((float3*)norm.pos)[tri.x];
	float3 v1 = ((float3*)norm.pos)[tri.y];
	float3 v2 = ((float3*)norm.pos)[tri.z];
	float3 n = normalize(cross(v1 - v0, v2 - v0));
	atomicAdd3(&((float3*)norm.out)[tri.x], n * acos(dot(v1 - v0, v2 - v0)));
	atomicAdd3(&((float3*)norm.out)[tri.y], n * acos(dot(v2 - v1, v0 - v1)));
	atomicAdd3(&((float3*)norm.out)[tri.z], n * acos(dot(v0 - v2, v1 - v2)));
}

__global__ void Normalize(const ComputeNormalKernelParams norm) {
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx >= norm.posNum)return;
	((float3*)norm.out)[pidx] = normalize(((float3*)norm.out)[pidx]);
}

void ComputeNormal::forward(ComputeNormalParams& norm) {
	CUDA_ERROR_CHECK(cudaMemset(norm.kernel.out, 0, norm.posSize()));
	void* args[] = { &norm.kernel };
	CUDA_ERROR_CHECK(cudaLaunchKernel(ComputeNormalForwardKernel, norm.vaogrid, norm.vaoblock, args, 0, NULL));
	CUDA_ERROR_CHECK(cudaLaunchKernel(Normalize, norm.grid, norm.block, args, 0, NULL));
}

void ComputeNormal::forward(ComputeNormalGradParams& norm) {
	CUDA_ERROR_CHECK(cudaMemset(norm.grad.out, 0, norm.posSize()));
	forward((ComputeNormalParams&)norm);
}

void ComputeNormal::init(ComputeNormalGradParams& norm, AttributeGrad& pos, AttributeGrad& normal) {
	init((ComputeNormalParams&)norm, pos, normal);
	norm.grad.out = normal.grad;
	norm.grad.pos = pos.grad;
}

__global__ void ComputeNormalBackwardKernel(const ComputeNormalKernelParams norm, const ComputeNormalKernelGradParams grad) {
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx >= norm.vaoNum)return;
}

void ComputeNormal::backward(ComputeNormalGradParams& norm) {
	void* args[] = { &norm.kernel,&norm.grad };
	CUDA_ERROR_CHECK(cudaLaunchKernel(ComputeNormalBackwardKernel, norm.grid, norm.block, args, 0, NULL));
}
