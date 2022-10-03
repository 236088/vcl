#include "extention.h"

void Noise::init(NoiseParams& np, RasterizeParams& rast, float* in, int channel, float intensity) {
	np.kernel.width = rast.kernel.width;
	np.kernel.height = rast.kernel.height;
	np.kernel.depth = rast.kernel.depth;
	np.kernel.channel = channel;
	np.kernel.intensity = intensity;
	np.kernel.in = in;
	CUDA_ERROR_CHECK(cudaMalloc(&np.kernel.out, np.Size()));
}

__global__ void NoiseForwardKernel(const NoiseKernelParams np, unsigned int seed) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= np.width || py >= np.height || pz >= np.depth)return;
	int pidx = px + np.width * (py + np.height * pz);
	for (int i = 0; i < np.channel; i++) {
		np.out[pidx * np.channel + i] = np.in[pidx * np.channel + i] + np.intensity * (getUniform(pidx, seed + i, 0xcafef00d) - np.in[pidx * np.channel + i]);
	}
}

void Noise::forward(NoiseParams& np) {
	CUDA_ERROR_CHECK(cudaMemset(np.kernel.out, 0, np.Size()));
	unsigned int seed = rand();
	dim3 block = getBlock(np.kernel.width, np.kernel.height);
	dim3  grid = getGrid(block, np.kernel.width, np.kernel.height);
	void* args[] = { &np.kernel ,&seed };
	CUDA_ERROR_CHECK(cudaLaunchKernel(NoiseForwardKernel, grid, block, args, 0, NULL));
}