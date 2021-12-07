#include "filter.h"

void Filter::init(FilterParams& flt, RasterizeParams& rast, float* in, int channel, int k) {
	flt.kernel.width = rast.kernel.width;
	flt.kernel.height = rast.kernel.height;
	flt.kernel.depth = rast.kernel.depth;
	flt.kernel.channel = channel;
	flt.kernel.k = k;
	flt.kernel.in = in;
	CUDA_ERROR_CHECK(cudaMalloc(&flt.kernel.out, flt.Size()));
	CUDA_ERROR_CHECK(cudaMalloc(&flt.kernel.buf, flt.Size()));
	flt.block = rast.block;
	flt.grid = rast.grid;
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
	void* args[] = { &flt.kernel };
	CUDA_ERROR_CHECK(cudaLaunchKernel(FilterForwardKernelHorizontal, flt.grid, flt.block, args, 0, NULL));
	CUDA_ERROR_CHECK(cudaLaunchKernel(FilterForwardKernelVertical, flt.grid, flt.block, args, 0, NULL));
}

void Filter::forward(FilterGradParams& flt){
	CUDA_ERROR_CHECK(cudaMemset(flt.grad.out, 0, flt.Size()));
	forward((FilterParams&)flt);
}

void Filter::init(FilterGradParams& flt, RasterizeParams& rast, float* in, float* grad, int channel, int k) {
	init((FilterParams&)flt, rast, in, channel, k);
	CUDA_ERROR_CHECK(cudaMalloc(&flt.grad.out, flt.Size()));
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
		}
	}
}

void Filter::backward(FilterGradParams& flt) {
	CUDA_ERROR_CHECK(cudaMemset(flt.grad.in, 0, flt.Size()));
	CUDA_ERROR_CHECK(cudaMemset(flt.kernel.buf, 0, flt.Size()));
	void* args[] = { &flt.kernel,&flt.grad };
	CUDA_ERROR_CHECK(cudaLaunchKernel(FilterBackwardKernelHorizontal, flt.grid, flt.block, args, 0, NULL));
	CUDA_ERROR_CHECK(cudaLaunchKernel(FilterBackwardKernelVertical, flt.grid, flt.block, args, 0, NULL));
}

void GaussianFilter::init(FilterParams& flt, RasterizeParams& rast, float* in, int channel, int k){
	Filter::init(flt, rast, in, channel, k);
	int num = 2 * k + 1;
	float* filter;
	CUDA_ERROR_CHECK(cudaMallocHost(&filter, (size_t)num * sizeof(float)));
	double d = exp2(-k * 2);
	filter[0] = filter[k * 2] = d;
	for (int s = 1, e = k * 2; s <= k;) {
		d *= double(e);
		d /= double(s);
		filter[s++] = d;
		filter[--e] = d;
	}
	CUDA_ERROR_CHECK(cudaMalloc(&flt.kernel.filter, (size_t)num * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMemcpy(flt.kernel.filter, filter, (size_t)num * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaFreeHost(filter));
}

void GaussianFilter::init(FilterGradParams& flt, RasterizeParams& rast, float* in, float* grad, int channel, int k) {
	init((FilterParams&)flt, rast, in, channel, k);
	CUDA_ERROR_CHECK(cudaMalloc(&flt.grad.out, flt.Size()));
	flt.grad.in = grad;
}

void MeanFilter::init(FilterParams& flt, RasterizeParams& rast, float* in, int channel, int k){
	Filter::init(flt, rast, in, channel, k);
	int num = 2 * k + 1;
	float* filter;
	CUDA_ERROR_CHECK(cudaMallocHost(&filter, (size_t)num * sizeof(float)));
	float n = 1.f / float(num);
	for (int i = 0; i < num; i++) {
		filter[i] = n;
	}
	CUDA_ERROR_CHECK(cudaMalloc(&flt.kernel.filter, (size_t)num * sizeof(float)));
	CUDA_ERROR_CHECK(cudaMemcpy(flt.kernel.filter, filter, (size_t)num * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_ERROR_CHECK(cudaFreeHost(filter));
}

void MeanFilter::init(FilterGradParams& flt, RasterizeParams& rast, float* in, float* grad, int channel, int k) {
	init((FilterParams&)flt, rast, in, channel, k);
	CUDA_ERROR_CHECK(cudaMalloc(&flt.grad.out, flt.Size()));
	flt.grad.in = grad;
}
