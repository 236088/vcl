#include "optimize.h"

void Optimizer::init(OptimizeParams& opt, float* param, float* grad, int size, int width, int height, int depth) {
	opt.kernel.param = param;
	opt.kernel.grad = grad;
	opt.kernel.size = size;
	opt.it = 1;
	opt.kernel.width = width;
	opt.kernel.height = height;
	opt.kernel.depth = depth;
	opt.block = getBlock(width, height);
	opt.grid = getGrid(opt.block, width, height, depth);
}

void Optimizer::init(OptimizeParams& opt, BufferGrad& buf) {
	int size = buf.num * buf.dimention;
	init(opt, buf.buffer, buf.grad, size, buf.num, buf.dimention, 1);
}

void Optimizer::init(OptimizeParams& opt, AttributeGrad& attr) {
	int size = attr.vboNum * attr.dimention;
	init(opt, attr.vbo, attr.grad, size, attr.vboNum, attr.dimention, 1);
}

void Optimizer::init(OptimizeParams& opt, TextureGrad& texture) {
	int size = texture.width * texture.height * texture.channel;
	init(opt, texture.texture[0], texture.grad[0], size, texture.width, texture.height, texture.channel);
}

__global__ void optimizeRandom(const OptimizeKernelParams opt, unsigned int seed, float min, float max) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + opt.width * (py + opt.height * pz);
	if (pidx >= opt.size)return;
	unsigned int rnd = pidx;
	opt.param[pidx] = min + (max - min) * getUniform(rnd, seed, 0xdeadbeef);
}

void Optimizer::randomParams(OptimizeParams& opt, float min, float max) {
	unsigned int seed = rand();
	void* args[] = { &opt.kernel ,&seed, &min, &max };
	CUDA_ERROR_CHECK(cudaLaunchKernel(optimizeRandom, opt.grid, opt.block, args, 0, NULL));
}

__global__ void optimizeClamp(const OptimizeKernelParams opt, unsigned int seed, float min, float max) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + opt.width * (py + opt.height * pz);
	if (pidx >= opt.size)return;
	unsigned int rnd = pidx;
	opt.param[pidx] = clamp(opt.param[pidx], min, max);
}

void Optimizer::clampParams(OptimizeParams& opt, float min, float max) {
	unsigned int seed = rand();
	void* args[] = { &opt.kernel ,&seed, &min, &max };
	CUDA_ERROR_CHECK(cudaLaunchKernel(optimizeClamp, opt.grid, opt.block, args, 0, NULL));
}



void SGD::setHyperParams(SGDParams& sgd, double eta) {
	sgd.hyper.eta = eta;
}

__global__ void sgdStep(const OptimizeKernelParams opt, const SGDHyperParams sgd) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + opt.width * (py + opt.height * pz);
	if (pidx >= opt.size)return;

	AddNaNcheck(opt.param[pidx], -sgd.eta * opt.grad[pidx]);
}

void SGD::step(SGDParams& sgd) {
	sgd.it++;
	void* args[] = {&sgd.kernel, &sgd.hyper };
	CUDA_ERROR_CHECK(cudaLaunchKernel(sgdStep, sgd.grid, sgd.block, args, 0, NULL));
}

void Adam::setHyperParams(AdamParams& adam, double eta, double rhom, double rhov, double eps) {
	adam.hyper.rhom = rhom;
	adam.hyper.rhov = rhov;
	adam.hyper.rhomt = rhom;
	adam.hyper.rhovt = rhov;
	adam.hyper.eta = eta;
	adam.hyper.eps = eps;
	CUDA_ERROR_CHECK(cudaMalloc(&adam.hyper.m, adam.Size()));
	CUDA_ERROR_CHECK(cudaMalloc(&adam.hyper.v, adam.Size()));
}

__global__ void adamStep(const OptimizeKernelParams opt, const AdamHyperParams adam) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + opt.width * (py + opt.height * pz);
	if (pidx >= opt.size)return;

	adam.m[pidx] = adam.rhom * adam.m[pidx] + (1 - adam.rhom) * opt.grad[pidx];
	adam.v[pidx] = adam.rhov * adam.v[pidx] + (1 - adam.rhov) * opt.grad[pidx] * opt.grad[pidx];
	double m = adam.m[pidx] / (1 - adam.rhomt);
	double v = adam.v[pidx] / (1 - adam.rhovt);
	AddNaNcheck(opt.param[pidx], -m * adam.eta / (sqrt(v) + adam.eps));
}

void Adam::step(AdamParams& adam) {
	adam.it++;
	adam.hyper.rhomt *= adam.hyper.rhom;
	adam.hyper.rhovt *= adam.hyper.rhov;
	void* args[] = {&adam.kernel, &adam.hyper };
	CUDA_ERROR_CHECK(cudaLaunchKernel(adamStep, adam.grid, adam.block, args, 0, NULL));
}


void Nadam::init(NadamParams& nadam, float* param, float* grad, int size, int width, int height, int depth) {
	Optimizer::init(nadam, param, grad, size, width, height, depth);
}

void Nadam::init(NadamParams& nadam, Attribute& attr, float* grad) {
	int size = attr.vboNum * attr.dimention;
	Optimizer::init(nadam, attr.vbo, grad, size, attr.vboNum, attr.dimention, 1);
}
void Nadam::setHyperParams(NadamParams& nadam, double alpha,  double mu, double rho, double eps) {
	nadam.hyper.mupow = pow(0.96, 0.004);
	nadam.hyper.mupowt = nadam.hyper.mupow;
	nadam.hyper.mu = mu;
	nadam.hyper.mu0 = mu * (1 - .5 * nadam.hyper.mupowt);
	nadam.hyper.mupowt *= nadam.hyper.mupow;
	nadam.hyper.mu1 = mu * (1 - .5 * nadam.hyper.mupowt);
	nadam.hyper.rho = rho;
	nadam.hyper.alpha = alpha;
	nadam.hyper.mut0 = nadam.hyper.mu0;
	nadam.hyper.mut1 = nadam.hyper.mu0 * nadam.hyper.mu1;
	nadam.hyper.rhot = rho;
	nadam.hyper.eps = eps;
	CUDA_ERROR_CHECK(cudaMalloc(&nadam.hyper.m, nadam.Size()));
	CUDA_ERROR_CHECK(cudaMalloc(&nadam.hyper.v, nadam.Size()));
}

__global__ void nadamStep(const OptimizeKernelParams opt, const NadamHyperParams nadam) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	int pidx = px + opt.width * (py + opt.height * pz);
	if (pidx >= opt.size)return;

	nadam.m[pidx] = nadam.mu * nadam.m[pidx] + (1 - nadam.mu) * opt.grad[pidx];
	nadam.v[pidx] = nadam.rho * nadam.v[pidx] + (1 - nadam.rho) * opt.grad[pidx] * opt.grad[pidx];
	double m =  nadam.mu1 / (1 - nadam.mut1) *nadam.m[pidx] + (1 - nadam.mu0) / (1- nadam.mut0) * opt.grad[pidx];
	double v = 1 / (1 - nadam.rhot) * nadam.v[pidx];
	AddNaNcheck(opt.param[pidx], -m * nadam.alpha / (sqrt(v) + nadam.eps));
}

void Nadam::step(NadamParams& nadam) {
	nadam.it++;
	nadam.hyper.mu0 = nadam.hyper.mu1;
	nadam.hyper.mut0 = nadam.hyper.mut1;
	nadam.hyper.mupowt *= nadam.hyper.mupow;
	nadam.hyper.mu1 = nadam.hyper.mu * (1 - .5 * nadam.hyper.mupowt);
	nadam.hyper.mut1 *= nadam.hyper.mu1;
	nadam.hyper.rhot *= nadam.hyper.rho;
	void* args[] = { &nadam.kernel, &nadam.hyper };
	CUDA_ERROR_CHECK(cudaLaunchKernel(nadamStep, nadam.grid, nadam.block, args, 0, NULL));
}