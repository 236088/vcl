#include "project.h"

void Project::init(ProjectParams& proj, float* mat, Attribute& vec, bool homogeneous) {
	if (vec.dimention != 3)ERROR_STRING(dimention is not 3);
	proj.kernel.vboNum = vec.vboNum;
	proj.kernel.dimention = homogeneous ? 4 : 3;
	proj.kernel.vbo = vec.vbo;
	proj.kernel.mat = mat;
	proj.vao = vec.vao;
	proj.vaoNum = vec.vaoNum;
	CUDA_ERROR_CHECK(cudaMalloc(&proj.kernel.out, proj.vboSize()));
}

void Project::init(ProjectParams& proj, float* mat, Attribute& vec, Attribute& out, bool homogeneous) {
	if (vec.dimention != 3)ERROR_STRING(dimention is not 3);
	Attribute::init(out, vec, homogeneous ? 4 : 3);
	proj.kernel.vboNum = vec.vboNum;
	proj.kernel.dimention = out.dimention;
	proj.kernel.vbo = vec.vbo;
	proj.kernel.mat = mat;
	proj.vao = vec.vao;
	proj.vaoNum = vec.vaoNum;
	proj.kernel.out = out.vbo;
}

__global__ void ProjectForwardKernel(const ProjectKernelParams proj) {
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx >= proj.vboNum)return;
	float3 v = ((float3*)proj.vbo)[pidx];
	for (int i = 0; i < proj.dimention; i++) {
		proj.out[pidx * proj.dimention + i] = proj.mat[i] * v.x + proj.mat[4 + i] * v.y + proj.mat[8 + i] * v.z + proj.mat[12 + i];
	}
}

void Project::forward(ProjectParams& proj) {
	dim3 block = getBlock(proj.kernel.vboNum, 1);
	dim3 grid = getGrid(block, proj.kernel.vboNum, 1);
	void* args[] = { &proj.kernel };
	CUDA_ERROR_CHECK(cudaLaunchKernel(ProjectForwardKernel, grid, block, args, 0, NULL));
}

void Project::forward(ProjectGradParams& proj) {
	CUDA_ERROR_CHECK(cudaMemset(proj.grad.out, 0, proj.vboSize()));
	forward((ProjectParams&)proj);
}

void Project::init(ProjectGradParams& proj, float* mat, AttributeGrad& vec, bool homogeneous) {
	init((ProjectParams&)proj, mat, vec, homogeneous);
	CUDA_ERROR_CHECK(cudaMalloc(&proj.grad.out, proj.vboSize()));
	proj.grad.vbo = vec.grad;
}

void Project::init(ProjectGradParams& proj, float* mat, AttributeGrad& vec, AttributeGrad& out, bool homogeneous) {
	init((ProjectParams&)proj, mat, vec, out, homogeneous);
	proj.grad.out = out.grad;
	proj.grad.vbo = vec.grad;
}

__global__ void ProjectBackwardKernel(const ProjectKernelParams proj, const ProjectKernelGradParams grad) {
	int pidx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pidx >= proj.vboNum)return;
	for (int i = 0; i < proj.dimention; i++) {
		grad.vbo[pidx * 3] += proj.mat[i] * grad.out[pidx * proj.dimention + i];
		grad.vbo[pidx * 3 + 1] += proj.mat[4 + i] * grad.out[pidx * proj.dimention + i];
		grad.vbo[pidx * 3 + 2] += proj.mat[8 + i] * grad.out[pidx * proj.dimention + i];
	}
}

void Project::backward(ProjectGradParams& proj) {
	dim3 block = getBlock(proj.kernel.vboNum, 1);
	dim3 grid = getGrid(block, proj.kernel.vboNum, 1);
	void* args[] = { &proj.kernel,&proj.grad };
	CUDA_ERROR_CHECK(cudaLaunchKernel(ProjectBackwardKernel, grid, block, args, 0, NULL));
}
