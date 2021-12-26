#include "common.h"
#include "texturemap.h"

void Texturemap::init(TexturemapParams& tex, RasterizeParams& rast, InterpolateParams& intr, Texture& texture) {
	tex.kernel.width = rast.kernel.width;
	tex.kernel.height = rast.kernel.height;
	tex.kernel.depth = rast.kernel.depth;
	tex.kernel.texwidth = texture.width;
	tex.kernel.texheight = texture.height;
	tex.kernel.channel = texture.channel;
	tex.kernel.miplevel = texture.miplevel;
	tex.kernel.rast = rast.kernel.out;
	tex.kernel.uv = intr.kernel.out;
	tex.kernel.uvDA = intr.kernel.outDA;

	CUDA_ERROR_CHECK(cudaMalloc(&tex.kernel.out, tex.Size()));
	for (int i = 0; i < texture.miplevel; i++) {
		tex.kernel.texture[i] = texture.texture[i];
	}
}

__device__ __forceinline__ void calculateLevel(const TexturemapKernelParams tex, int pidx, int& level0, int& level1, float& flevel) {
	float4 uvDA = ((float4*)tex.uvDA)[pidx];
	float dsdx = uvDA.x * tex.texwidth;
	float dsdy = uvDA.y * tex.texwidth;
	float dtdx = uvDA.z * tex.texheight;
	float dtdy = uvDA.w * tex.texheight;

	// calculate footprint
	// b is sum of 2 square sides 
	// b = (dsdx^2+dsdy^2) + (dtdx^2+dtdy^2)
	// c is square area
	// c = (dsdx * dtdy - dtdx * dsdy)^2
	// solve x^2 - bx + c = 0

	float s2 = dsdx * dsdx + dsdy * dsdy;
	float t2 = dtdx * dtdx + dtdy * dtdy;
	float a = dsdx * dtdy - dtdx * dsdy;

	float b = .5f * (s2 + t2);
	float c = sqrt(b * b - a * a);

	float level = .5f * log2f(b + c);
	level0 = level <= 0 ? 0 : tex.miplevel - 2 <= level ? tex.miplevel - 2 : (int)floor(level);
	level1 = level <= 1 ? 1 : tex.miplevel - 1 <= level ? tex.miplevel - 1 : (int)floor(level) + 1;
	flevel = level <= 0 ? 0 : tex.miplevel - 1 <= level ? 1 : level - floor(level);
}

__device__ __forceinline__ int4 indexFetch(const TexturemapKernelParams tex, int level, float2 uv, float2& t) {
	int2 size = make_int2(tex.texwidth >> level, tex.texheight >> level);
	t.x = uv.x * (float)size.x;
	t.y = uv.y * (float)size.y;
	int u0 = t.x<0 ? 0 : t.x>size.x - 1 ? size.x - 1 : (int)t.x;
	int u1 = t.x<1 ? 0 : t.x>size.x - 2 ? size.x - 1 : (int)t.x + 1;
	int v0 = t.y<0 ? 0 : t.y>size.y - 1 ? size.y - 1 : (int)t.y;
	int v1 = t.y<1 ? 0 : t.y>size.y - 2 ? size.y - 1 : (int)t.y + 1;
	int4 idx;
	idx.x = v0 * size.x + u0;
	idx.y = v0 * size.x + u1;
	idx.z = v1 * size.x + u0;
	idx.w = v1 * size.x + u1;
	t.x = t.x<0 ? 0 : size.x<t.x ? 1 : t.x - floor(t.x);
	t.y = t.y<0 ? 0 : size.y<t.y ? 1 : t.y - floor(t.y);
	return idx;
}

__global__ void TexturemapMipForwardKernel(const TexturemapKernelParams tex) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= tex.width || py >= tex.height || pz >= tex.depth)return;
	int pidx = px + tex.width * (py + tex.height * pz);

	if (tex.rast[pidx * 4 + 3] < 1.f) return;

	int level0, level1;
	float flevel;
	calculateLevel(tex, pidx, level0, level1, flevel);
	float2 uv = ((float2*)tex.uv)[pidx];
	float2 uv0, uv1;
	int4 idx0 = indexFetch(tex, level0, uv, uv0);
	int4 idx1 = indexFetch(tex, level1, uv, uv1);
	for (int i = 0; i < tex.channel; i++) {
		float out = bilerp(
			tex.texture[level0][idx0.x * tex.channel + i], tex.texture[level0][idx0.y * tex.channel + i],
			tex.texture[level0][idx0.z * tex.channel + i], tex.texture[level0][idx0.w * tex.channel + i], uv0);
		if (flevel > 0) {
			float out1 = bilerp(
				tex.texture[level1][idx1.x * tex.channel + i], tex.texture[level1][idx1.y * tex.channel + i],
				tex.texture[level1][idx1.z * tex.channel + i], tex.texture[level1][idx1.w * tex.channel + i], uv1);
			out = lerp(out, out1, flevel);
		}
		tex.out[pidx * tex.channel + i] = out;
	}
}

__global__ void TexturemapNoMipForwardKernel(const TexturemapKernelParams tex) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= tex.width || py >= tex.height || pz >= tex.depth)return;
	int pidx = px + tex.width * (py + tex.height * pz);

	if (tex.rast[pidx * 4 + 3] < 1.f) return;

	float2 uv = ((float2*)tex.uv)[pidx];
	float2 uv0;
	int4 idx0 = indexFetch(tex, 0, uv, uv0);
	for (int i = 0; i < tex.channel; i++) {
		tex.out[pidx * tex.channel + i] = bilerp(
			tex.texture[0][idx0.x * tex.channel + i], tex.texture[0][idx0.y * tex.channel + i],
			tex.texture[0][idx0.z * tex.channel + i], tex.texture[0][idx0.w * tex.channel + i], uv0);
	}
}

void Texturemap::forward(TexturemapParams& tex) {
	CUDA_ERROR_CHECK(cudaMemset(tex.kernel.out, 0, tex.Size()));
	void* args[] = { &tex.kernel };
	dim3 block = getBlock(tex.kernel.width, tex.kernel.height);
	dim3 grid = getGrid(block, tex.kernel.width, tex.kernel.height, tex.kernel.depth);
	if (tex.kernel.miplevel > 1) CUDA_ERROR_CHECK(cudaLaunchKernel(TexturemapMipForwardKernel, grid,block, args, 0, NULL));
	else CUDA_ERROR_CHECK(cudaLaunchKernel(TexturemapNoMipForwardKernel, grid, block, args, 0, NULL));
}

void Texturemap::forward(TexturemapGradParams& tex) {
	CUDA_ERROR_CHECK(cudaMemset(tex.grad.out, 0, tex.Size()));
	forward((TexturemapParams&)tex);
}

void Texturemap::init(TexturemapGradParams& tex, RasterizeParams& rast, InterpolateGradParams& intr, TextureGrad& texture) {
	init((TexturemapParams&)tex, rast, intr, texture);
	CUDA_ERROR_CHECK(cudaMalloc(&tex.grad.out, tex.Size()));
	tex.grad.uv = intr.grad.out;
	tex.grad.uvDA = intr.grad.outDA;
	for (int i = 0; i < texture.miplevel; i++) {
		tex.grad.texture[i] = texture.grad[i];
	}
}

void Texturemap::init(TexturemapGradParams& tex, RasterizeParams& rast, InterpolateParams& intr, TextureGrad& texture) {
	init((TexturemapParams&)tex, rast, intr, texture);
	CUDA_ERROR_CHECK(cudaMalloc(&tex.grad.out, tex.Size()));
	tex.grad.uv = nullptr;
	tex.grad.uvDA = nullptr;
	for (int i = 0; i < texture.miplevel; i++) {
		tex.grad.texture[i] = texture.grad[i];
	}
}

void Texturemap::init(TexturemapGradParams& tex, RasterizeParams& rast, InterpolateGradParams& intr, Texture& texture) {
	init((TexturemapParams&)tex, rast, intr, texture);
	CUDA_ERROR_CHECK(cudaMalloc(&tex.grad.out, tex.Size()));
	tex.grad.uv = intr.grad.out;
	tex.grad.uvDA = intr.grad.outDA;
	for (int i = 0; i < texture.miplevel; i++) {
		tex.grad.texture[i] = nullptr;
	}
}

__device__ __forceinline__ void calculateLevelWithJacob(const TexturemapKernelParams tex, int pidx, int& level0, int& level1, float& flevel, float4& dleveldda) {
	float4 uvDA = ((float4*)tex.uvDA)[pidx];
	float dsdx = uvDA.x * tex.texwidth;
	float dsdy = uvDA.y * tex.texwidth;
	float dtdx = uvDA.z * tex.texheight;
	float dtdy = uvDA.w * tex.texheight;

	float s2 = dsdx * dsdx + dsdy * dsdy;
	float t2 = dtdx * dtdx + dtdy * dtdy;
	float a = dsdx * dtdy - dtdx * dsdy;

	float b = .5f * (s2 + t2);
	float c2 = b * b - a * a;
	float c = sqrt(c2);


	float level = .5f * log2f(b + c);
	level0 = level <= 0 ? 0 : tex.miplevel - 2 <= level ? tex.miplevel - 2 : (int)floor(level);
	level1 = level <= 1 ? 1 : tex.miplevel - 1 <= level ? tex.miplevel - 1 : (int)floor(level) + 1;
	flevel = level <= 0 ? 0 : tex.miplevel - 1 <= level ? 1 : level - floor(level);

	float d = b * c + c2; // b^2 - a^2 == 0 or not if 0 then level=ln(b)
	if (abs(d) > 1e-6) {
		d = 0.72134752f / d;
		float bc = b + c;
		dleveldda = make_float4(d * (bc * dsdx - a * dtdy), d * (bc * dsdy + a * dtdx), (bc * dtdx + a * dsdy), (bc * dtdy - a * dsdx));
	}
	else {
		// if abs(b) == 0 then dsdx, dsdy, dtdx, dtdy are 0
		if (abs(b) > 1e-6) {
			d = 1 / b;
			dleveldda = make_float4(d * dsdx, d * dsdy, d * dtdx, d * dtdy);
		}
		else {
			dleveldda = make_float4(0.f, 0.f, 0.f, 0.f);
		}	
	}
}

// s_ = frac(s*width_) => d/ds = d/ds_ * width_
// t_ = frac(t*height_) => d/dt = d/dt_ * height_
// l = frac(level) => dl/dlevel = 1
//
// dL/dX = dL/dc * dc/dX
//
// dc/ds = lerp(lerp(c001-c000, c011-c010, t0) * width0, lerp(c101-c100, c111-c110, t1) * width1, l)
// dc/dt = lerp(lerp(c010-c000, c011-c001, s0) * height0, lerp(c110-c100, c111-c101, s1) * height1, l)
// dc/dlevel = -bilerp(c000,c001,c010,c011,s0,t0) + bilerp(c100,c101,c110,c111,s1,t1)
//
// dc/dc000 = (1-l) * (1-s0) * (1-t0)
// :
// :
// dc/dc111 = l * s1 * t1
// 
// 
//
// dL/dX = dL/dc * dc/dlevel * dlevel/dX
// 
// b = ((ds/dx^2+ds/dy^2) + (dt/dx^2+dt/dy^2))/2
// a = ds/dx * dt/dy - dt/dx * ds/dy
// level = ln(b + sqrt(b^2 - a^2))/2ln2
//
// dlevel/dX = 1/2ln2 * (b'+(b*b'-a*a')/sqrt(b^2-a^2))/(b+sqrt(b^2-a^2))
//           = 1/2ln2/(b * sqrt(b^2-a^2) + (b^2-a^2)) * ((sqrt(b^2-a^2) + b) * b'- a * a')
// dlevel/d(ds/dx) = 1/2ln2/(b * sqrt(b^2-a^2) + (b^2-a^2)) * ((sqrt(b^2-a^2) + b) * ds/dx - a * dt/dy)
// dlevel/d(ds/dy) = 1/2ln2/(b * sqrt(b^2-a^2) + (b^2-a^2)) * ((sqrt(b^2-a^2) + b) * ds/dy + a * dt/dx)
// dlevel/d(dt/dx) = 1/2ln2/(b * sqrt(b^2-a^2) + (b^2-a^2)) * ((sqrt(b^2-a^2) + b) * dt/dx + a * ds/dy)
// dlevel/d(dt/dy) = 1/2ln2/(b * sqrt(b^2-a^2) + (b^2-a^2)) * ((sqrt(b^2-a^2) + b) * dt/dy - a * ds/dx)
//
__global__ void TexturemapMipBackwardKernel(const TexturemapKernelParams tex, const TexturemapKernelGradParams grad) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= tex.width || py >= tex.height || pz >= tex.depth)return;
	int pidx = px + tex.width * (py + tex.height * pz);
	if (tex.rast[pidx * 4 + 3] < 1.f) return;

	int level0 = 0, level1 = 0;
	float flevel = 0.f;
	float4 dleveldda;
	calculateLevelWithJacob(tex, pidx, level0, level1, flevel, dleveldda);
	float2 uv = ((float2*)tex.uv)[pidx], uv0, uv1;
	float gu = 0.f, gv = 0.f, gl = 0.f;
	int4 idx0 = indexFetch(tex, level0, uv, uv0);
	int4 idx1 = indexFetch(tex, level1, uv, uv1);

	for (int i = 0; i < tex.channel; i++) {
		float dLdout = grad.out[pidx * tex.channel + i];
		if (grad.texture != nullptr) {
			atomicAdd(&grad.texture[level0][idx0.x * tex.channel + i], (1.f - flevel) * (1.f - uv0.x) * (1.f - uv0.y) * dLdout);
			atomicAdd(&grad.texture[level0][idx0.y * tex.channel + i], (1.f - flevel) * uv0.x * (1.f - uv0.y) * dLdout);
			atomicAdd(&grad.texture[level0][idx0.z * tex.channel + i], (1.f - flevel) * (1.f - uv0.x) * uv0.y * dLdout);
			atomicAdd(&grad.texture[level0][idx0.w * tex.channel + i], (1.f - flevel) * uv0.x * uv0.y * dLdout);
		}
		float t00 = tex.texture[level0][idx0.x * tex.channel + i];
		float t01 = tex.texture[level0][idx0.y * tex.channel + i];
		float t10 = tex.texture[level0][idx0.z * tex.channel + i];
		float t11 = tex.texture[level0][idx0.w * tex.channel + i];

		float u = lerp(t01 - t00, t11 - t10, uv0.y) * (tex.texwidth >> level0);
		float v = lerp(t10 - t00, t11 - t01, uv0.x) * (tex.texheight >> level0);
		if (flevel > 0 && tex.miplevel > 1) {
			float l = bilerp(t00, t01, t10, t11, uv0);
			if (grad.texture != nullptr) {
				atomicAdd(&grad.texture[level1][idx1.x * tex.channel + i], flevel * (1.f - uv1.x) * (1.f - uv1.y) * dLdout);
				atomicAdd(&grad.texture[level1][idx1.y * tex.channel + i], flevel * uv1.x * (1.f - uv1.y) * dLdout);
				atomicAdd(&grad.texture[level1][idx1.z * tex.channel + i], flevel * (1.f - uv1.x) * uv1.y * dLdout);
				atomicAdd(&grad.texture[level1][idx1.w * tex.channel + i], flevel * uv1.x * uv1.y * dLdout);
			}
			t00 = tex.texture[level1][idx1.x * tex.channel + i];
			t01 = tex.texture[level1][idx1.y * tex.channel + i];
			t10 = tex.texture[level1][idx1.z * tex.channel + i];
			t11 = tex.texture[level1][idx1.w * tex.channel + i];
			u = lerp(u, lerp(t01 - t00, t11 - t10, uv1.y) * (tex.texwidth >> level1), flevel);
			v = lerp(v, lerp(t10 - t00, t11 - t01, uv1.x) * (tex.texheight >> level1), flevel);
			gl += (bilerp(t00, t01, t10, t11, uv1) - l) * dLdout;
		}
		gu += u * dLdout;
		gv += v * dLdout;
	}
	if (grad.uv != nullptr) {
		((float2*)grad.uv)[pidx] = make_float2(gu, gv);
		((float4*)grad.uvDA)[pidx] = gl * dleveldda;
	}
}

__global__ void TexturemapNoMipBackwardKernel(const TexturemapKernelParams tex, const TexturemapKernelGradParams grad) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= tex.width || py >= tex.height || pz >= tex.depth)return;
	int pidx = px + tex.width * (py + tex.height * pz);
	if (tex.rast[pidx * 4 + 3] < 1.f) return;

	float2 uv = ((float2*)tex.uv)[pidx], uv0, uv1;
	float gu = 0.f, gv = 0.f, gl = 0.f;
	int4 idx0 = indexFetch(tex, 0, uv, uv0);

	for (int i = 0; i < tex.channel; i++) {
		float dLdout = grad.out[pidx * tex.channel + i];
		if (grad.texture != nullptr) {
			atomicAdd(&grad.texture[0][idx0.x * tex.channel + i], (1.f - uv0.x) * (1.f - uv0.y) * dLdout);
			atomicAdd(&grad.texture[0][idx0.y * tex.channel + i], uv0.x * (1.f - uv0.y) * dLdout);
			atomicAdd(&grad.texture[0][idx0.z * tex.channel + i], (1.f - uv0.x) * uv0.y * dLdout);
			atomicAdd(&grad.texture[0][idx0.w * tex.channel + i], uv0.x * uv0.y * dLdout);
		}
		float t00 = tex.texture[0][idx0.x * tex.channel + i];
		float t01 = tex.texture[0][idx0.y * tex.channel + i];
		float t10 = tex.texture[0][idx0.z * tex.channel + i];
		float t11 = tex.texture[0][idx0.w * tex.channel + i];

		float u = lerp(t01 - t00, t11 - t10, uv0.y) * (tex.texwidth);
		float v = lerp(t10 - t00, t11 - t01, uv0.x) * (tex.texheight);
		gu += u * dLdout;
		gv += v * dLdout;
	}
	if (grad.uv != nullptr) {
		((float2*)grad.uv)[pidx] = make_float2(gu, gv);
	}
}

void Texturemap::backward(TexturemapGradParams& tex) {
	dim3 block = getBlock(tex.kernel.width, tex.kernel.height);
	dim3 grid = getGrid(block, tex.kernel.width, tex.kernel.height, tex.kernel.depth);
	void* args[] = { &tex.kernel, &tex.grad };
	if(tex.kernel.miplevel>1)CUDA_ERROR_CHECK(cudaLaunchKernel(TexturemapMipBackwardKernel, grid, block, args, 0, NULL));
	else CUDA_ERROR_CHECK(cudaLaunchKernel(TexturemapNoMipBackwardKernel, grid, block, args, 0, NULL));
}