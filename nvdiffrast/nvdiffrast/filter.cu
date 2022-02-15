#include "filter.h"
#define FILTER_MAX_SIZE 255

void Filter::init(FilterParams& flt, RasterizeParams& rast, float* in, int channel, float sigma) {
	flt.kernel.width = rast.kernel.width;
	flt.kernel.height = rast.kernel.height;
	flt.kernel.depth = rast.kernel.depth;
	flt.kernel.channel = channel;
	flt.kernel.in = in;
	flt.h_sig = sigma;
	CUDA_ERROR_CHECK(cudaMalloc(&flt.kernel.sigma, sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&flt.kernel.filter, FILTER_MAX_SIZE * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMemcpy(flt.kernel.sigma, &sigma, sizeof(float), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaMalloc(&flt.kernel.out, flt.Size()));
	CUDA_ERROR_CHECK(cudaMalloc(&flt.kernel.buf, flt.Size()));
}

__device__ __forceinline__ void calculateRange(int p, int k, int w, int& s, int& e, int& f) {
	s = p - k;
	f = 0;
	if (s < 0) {
		f = -s; s = 0;
	}
	e = p + k;
	if (e >= w) {
		e = w - 1;
	}
}

__global__ void FilterForwardKernelHorizontal(const FilterKernelParams flt) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= flt.width || py >= flt.height || pz >= flt.depth)return;
	int pidx = px + flt.width * (py + flt.height * pz);
	int s, e, f;
	calculateRange(px, flt.k, flt.width, s, e, f);
	for (int i = s; i <= e; i++, f++) {
		for (int k = 0; k < flt.channel; k++) {
			int idx = i + flt.width * py;
			flt.buf[pidx * flt.channel + k] += flt.in[idx * flt.channel + k] * flt.filter[f];
		}
	}
}

__global__ void FilterForwardKernelVertical(const FilterKernelParams flt) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= flt.width || py >= flt.height || pz >= flt.depth)return;
	int pidx = px + flt.width * (py + flt.height * pz);
	int s, e, f;
	calculateRange(py, flt.k, flt.height, s, e, f);
	for (int i = s; i <= e; i++, f++) {
		for (int k = 0; k < flt.channel; k++) {
			int idx = px + flt.height * i;
			flt.out[pidx * flt.channel + k] += flt.buf[idx * flt.channel + k] * flt.filter[f];
		}
	}
}

void Filter::forward(FilterParams& flt) {
	CUDA_ERROR_CHECK(cudaMemset(flt.kernel.out, 0, flt.Size()));
	CUDA_ERROR_CHECK(cudaMemset(flt.kernel.buf, 0, flt.Size()));
	CUDA_ERROR_CHECK(cudaMemcpy(&flt.h_sig, flt.kernel.sigma, sizeof(float), cudaMemcpyDeviceToHost));
	int k = int(ceil(flt.h_sig * 4.f));
	if (k * 2 > FILTER_MAX_SIZE)k = FILTER_MAX_SIZE / 2;
	flt.kernel.k = k;
	float* filter;
	CUDA_ERROR_CHECK(cudaMallocHost(&filter, (2 * k + 1) * sizeof(float)));
	float s = 0.f;
	float s2 = .5f / (flt.h_sig * flt.h_sig);
	for (int i = -k; i <= k; i++) {
		filter[i + k] = exp(-i * i * s2);
		s += filter[i + k];
	}
	s = 1.f / s;
	for (int i = -k; i <= k; i++) {
		filter[i + k] *= s;
	}
	CUDA_ERROR_CHECK(cudaMemcpy(flt.kernel.filter, filter, (2 * k + 1) * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaFreeHost(filter));
	dim3 block = getBlock(flt.kernel.width, flt.kernel.height);
	dim3 grid = getGrid(block, flt.kernel.width, flt.kernel.height, flt.kernel.depth);
	void* args[] = { &flt.kernel };
	CUDA_ERROR_CHECK(cudaLaunchKernel(FilterForwardKernelHorizontal, grid, block, args, 0, NULL));
	CUDA_ERROR_CHECK(cudaLaunchKernel(FilterForwardKernelVertical, grid, block, args, 0, NULL));
}

void Filter::forward(FilterGradParams& flt) {
	CUDA_ERROR_CHECK(cudaMemset(flt.grad.out, 0, flt.Size()));
	forward((FilterParams&)flt);
}

void Filter::init(FilterGradParams& flt, RasterizeParams& rast, float* in, float* grad, int channel, float sigma) {
	init((FilterParams&)flt, rast, in, channel, sigma);
	CUDA_ERROR_CHECK(cudaMalloc(&flt.grad.sigma, sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&flt.grad.filter, FILTER_MAX_SIZE * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMalloc(&flt.grad.out, flt.Size()));
	CUDA_ERROR_CHECK(cudaMalloc(&flt.grad.buf, flt.Size()));
	CUDA_ERROR_CHECK(cudaMalloc(&flt.grad.bufx, flt.Size()));
	CUDA_ERROR_CHECK(cudaMalloc(&flt.grad.bufy, flt.Size()));
	flt.grad.in = grad;
}

__global__ void FilterBackwardKernelHorizontal(const FilterKernelParams flt, const FilterKernelGradParams grad) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= flt.width || py >= flt.height || pz >= flt.depth)return;
	int pidx = px + flt.width * (py + flt.height * pz);
	int s, e, f;
	calculateRange(px, flt.k, flt.width, s, e, f);
	for (int i = s; i <= e; i++, f++) {
		for (int k = 0; k < flt.channel; k++) {
			int idx = i + flt.width * py;
			flt.buf[pidx * flt.channel + k] += grad.out[idx * flt.channel + k] * flt.filter[f];
			grad.buf[pidx * flt.channel + k] += grad.out[idx * flt.channel + k] * grad.filter[f];
		}
	}
}

__global__ void FilterBackwardKernelVertical(const FilterKernelParams flt, const FilterKernelGradParams grad) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= flt.width || py >= flt.height || pz >= flt.depth)return;
	int pidx = px + flt.width * (py + flt.height * pz);
	int s, e, f;
	calculateRange(py, flt.k, flt.height, s, e, f);
	for (int i = s; i <= e; i++, f++) {
		for (int k = 0; k < flt.channel; k++) {
			int idx = px + flt.height * i;
			grad.in[pidx * flt.channel + k] += flt.buf[idx * flt.channel + k] * flt.filter[f];
			grad.bufx[pidx * flt.channel + k] += flt.buf[idx * flt.channel + k] * grad.filter[f];
			grad.bufy[pidx * flt.channel + k] += grad.buf[idx * flt.channel + k] * flt.filter[f];
		}
	}
}

__global__ void FilterBackwardKernelFinal(const FilterKernelParams flt, const FilterKernelGradParams grad, float is3, float is2) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= flt.width || py >= flt.height || pz >= flt.depth)return;
	int pidx = px + flt.width * (py + flt.height * pz);
	int s, e, f;
	calculateRange(py, flt.k, flt.height, s, e, f);
	for (int i = s; i <= e; i++, f++) {
		for (int k = 0; k < flt.channel; k++) {
			int idx = px + flt.height * i;
			grad.buf[pidx * flt.channel + k] = (grad.bufx[pidx * flt.channel + k] + grad.bufy[pidx * flt.channel + k]) * is3 - grad.in[pidx * flt.channel + k] * is2;
			grad.buf[pidx * flt.channel + k] *= flt.in[pidx * flt.channel + k];
			atomicAdd(grad.sigma, grad.buf[pidx * flt.channel + k]);
		}
	}
}

void Filter::backward(FilterGradParams& flt) {
	CUDA_ERROR_CHECK(cudaMemset(flt.grad.in, 0, flt.Size()));
	CUDA_ERROR_CHECK(cudaMemset(flt.grad.sigma, 0, sizeof(float)));
	CUDA_ERROR_CHECK(cudaMemset(flt.kernel.buf, 0, flt.Size()));
	CUDA_ERROR_CHECK(cudaMemset(flt.grad.buf, 0, flt.Size()));
	CUDA_ERROR_CHECK(cudaMemset(flt.grad.bufx, 0, flt.Size()));
	CUDA_ERROR_CHECK(cudaMemset(flt.grad.bufy, 0, flt.Size()));
	CUDA_ERROR_CHECK(cudaMemcpy(&flt.h_sig, flt.kernel.sigma, sizeof(float), cudaMemcpyDeviceToHost));
	int k = flt.kernel.k;
	float* filter;
	CUDA_ERROR_CHECK(cudaMallocHost(&filter, (2 * k + 1) * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMemcpy(filter, flt.kernel.filter, (2 * k + 1) * sizeof(float), cudaMemcpyDeviceToHost));
	float s2 = .5f / (flt.h_sig * flt.h_sig);
	for (int i = -k; i <= k; i++) {
		filter[i + k] *= i * i;
	}
	CUDA_ERROR_CHECK(cudaMemcpy(flt.grad.filter, filter, (2 * k + 1) * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaFreeHost(filter));
	void* args[] = { &flt.kernel,&flt.grad };
	dim3 block = getBlock(flt.kernel.width, flt.kernel.height);
	dim3 grid = getGrid(block, flt.kernel.width, flt.kernel.height, flt.kernel.depth);
	CUDA_ERROR_CHECK(cudaLaunchKernel(FilterBackwardKernelHorizontal, grid, block, args, 0, NULL));
	CUDA_ERROR_CHECK(cudaLaunchKernel(FilterBackwardKernelVertical, grid, block, args, 0, NULL));
	float is2 = 2.f / flt.h_sig;
	float is3 = s2 * is2;
	void* fargs[] = { &flt.kernel,&flt.grad ,&is3,&is2 };
	CUDA_ERROR_CHECK(cudaLaunchKernel(FilterBackwardKernelFinal, grid, block, fargs, 0, NULL));
}