#pragma once
#include <GL/glew.h>
#include <GL/glut.h>

#include <device_launch_parameters.h>
#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#define __CUDACC_VER_MAJOR__ 11
#define __CUDACC_VER_MINOR__ 4
#include <device_atomic_functions.h>
#include <cuda_runtime.h>

#include <math.h>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <vector>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
using namespace std;

#define MAX_DIM_PER_BLOCK 32

dim3 getBlock(int width, int height);
dim3 getGrid(dim3 block, int width, int height);
dim3 getGrid(dim3 block, int width, int height, int depth);
int MSB(size_t v);
int LSB(size_t v);

#define ERROR_STRING(str) do{\
std::cout<<#str<<" in file "<<__FILE__<<" at line "<<__LINE__<<std::endl;\
exit(1);\
}while(0)

#define CUDA_ERROR_CHECK(call) do{\
cudaError_t e = call;\
if (e != cudaSuccess) {\
std::cout<<cudaGetErrorString(e)<<" in file "<<__FILE__<<" at line "<<__LINE__<<std::endl;\
exit(1);\
}\
}while(0)

static __device__ __forceinline__ float2& operator*=  (float2& a, const float2& b) { a.x *= b.x; a.y *= b.y; return a; }
static __device__ __forceinline__ float2& operator+=  (float2& a, const float2& b) { a.x += b.x; a.y += b.y; return a; }
static __device__ __forceinline__ float2& operator-=  (float2& a, const float2& b) { a.x -= b.x; a.y -= b.y; return a; }
static __device__ __forceinline__ float2& operator*=  (float2& a, float b) { a.x *= b; a.y *= b; return a; }
static __device__ __forceinline__ float2& operator+=  (float2& a, float b) { a.x += b; a.y += b; return a; }
static __device__ __forceinline__ float2& operator-=  (float2& a, float b) { a.x -= b; a.y -= b; return a; }
static __device__ __forceinline__ float2    operator*   (const float2& a, const float2& b) { return make_float2(a.x * b.x, a.y * b.y); }
static __device__ __forceinline__ float2    operator+   (const float2& a, const float2& b) { return make_float2(a.x + b.x, a.y + b.y); }
static __device__ __forceinline__ float2    operator-   (const float2& a, const float2& b) { return make_float2(a.x - b.x, a.y - b.y); }
static __device__ __forceinline__ float2    operator*   (const float2& a, float b) { return make_float2(a.x * b, a.y * b); }
static __device__ __forceinline__ float2    operator+   (const float2& a, float b) { return make_float2(a.x + b, a.y + b); }
static __device__ __forceinline__ float2    operator-   (const float2& a, float b) { return make_float2(a.x - b, a.y - b); }
static __device__ __forceinline__ float2    operator*   (float a, const float2& b) { return make_float2(a * b.x, a * b.y); }
static __device__ __forceinline__ float2    operator+   (float a, const float2& b) { return make_float2(a + b.x, a + b.y); }
static __device__ __forceinline__ float2    operator-   (float a, const float2& b) { return make_float2(a - b.x, a - b.y); }
static __device__ __forceinline__ float2    operator-   (const float2& a) { return make_float2(-a.x, -a.y); }
static __device__ __forceinline__ float3& operator*=  (float3& a, const float3& b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; return a; }
static __device__ __forceinline__ float3& operator+=  (float3& a, const float3& b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
static __device__ __forceinline__ float3& operator-=  (float3& a, const float3& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; return a; }
static __device__ __forceinline__ float3& operator*=  (float3& a, float b) { a.x *= b; a.y *= b; a.z *= b; return a; }
static __device__ __forceinline__ float3& operator+=  (float3& a, float b) { a.x += b; a.y += b; a.z += b; return a; }
static __device__ __forceinline__ float3& operator-=  (float3& a, float b) { a.x -= b; a.y -= b; a.z -= b; return a; }
static __device__ __forceinline__ float3    operator*   (const float3& a, const float3& b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
static __device__ __forceinline__ float3    operator+   (const float3& a, const float3& b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
static __device__ __forceinline__ float3    operator-   (const float3& a, const float3& b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
static __device__ __forceinline__ float3    operator*   (const float3& a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }
static __device__ __forceinline__ float3    operator+   (const float3& a, float b) { return make_float3(a.x + b, a.y + b, a.z + b); }
static __device__ __forceinline__ float3    operator-   (const float3& a, float b) { return make_float3(a.x - b, a.y - b, a.z - b); }
static __device__ __forceinline__ float3    operator*   (float a, const float3& b) { return make_float3(a * b.x, a * b.y, a * b.z); }
static __device__ __forceinline__ float3    operator+   (float a, const float3& b) { return make_float3(a + b.x, a + b.y, a + b.z); }
static __device__ __forceinline__ float3    operator-   (float a, const float3& b) { return make_float3(a - b.x, a - b.y, a - b.z); }
static __device__ __forceinline__ float3    operator-   (const float3& a) { return make_float3(-a.x, -a.y, -a.z); }
static __device__ __forceinline__ float4& operator*=  (float4& a, const float4& b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w; return a; }
static __device__ __forceinline__ float4& operator+=  (float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; return a; }
static __device__ __forceinline__ float4& operator-=  (float4& a, const float4& b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w; return a; }
static __device__ __forceinline__ float4& operator*=  (float4& a, float b) { a.x *= b; a.y *= b; a.z *= b; a.w *= b; return a; }
static __device__ __forceinline__ float4& operator+=  (float4& a, float b) { a.x += b; a.y += b; a.z += b; a.w += b; return a; }
static __device__ __forceinline__ float4& operator-=  (float4& a, float b) { a.x -= b; a.y -= b; a.z -= b; a.w -= b; return a; }
static __device__ __forceinline__ float4    operator*   (const float4& a, const float4& b) { return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
static __device__ __forceinline__ float4    operator+   (const float4& a, const float4& b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
static __device__ __forceinline__ float4    operator-   (const float4& a, const float4& b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
static __device__ __forceinline__ float4    operator*   (const float4& a, float b) { return make_float4(a.x * b, a.y * b, a.z * b, a.w * b); }
static __device__ __forceinline__ float4    operator+   (const float4& a, float b) { return make_float4(a.x + b, a.y + b, a.z + b, a.w + b); }
static __device__ __forceinline__ float4    operator-   (const float4& a, float b) { return make_float4(a.x - b, a.y - b, a.z - b, a.w - b); }
static __device__ __forceinline__ float4    operator*   (float a, const float4& b) { return make_float4(a * b.x, a * b.y, a * b.z, a * b.w); }
static __device__ __forceinline__ float4    operator+   (float a, const float4& b) { return make_float4(a + b.x, a + b.y, a + b.z, a + b.w); }
static __device__ __forceinline__ float4    operator-   (float a, const float4& b) { return make_float4(a - b.x, a - b.y, a - b.z, a - b.w); }

static __device__ __forceinline__ float dot(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }
static __device__ __forceinline__ float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static __device__ __forceinline__ float dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
static __device__ __forceinline__ float length(float2 v) { return sqrt(dot(v, v)); }
static __device__ __forceinline__ float length(float3 v) { return sqrt(dot(v, v)); }
static __device__ __forceinline__ float length(float4 v) { return sqrt(dot(v, v)); }
static __device__ __forceinline__ float2 normalize(float2 v) { float d = 1.f / length(v); v.x *= d; v.y *= d; return v; }
static __device__ __forceinline__ float3 normalize(float3 v) { float d = 1.f / length(v); v.x *= d; v.y *= d; v.z *= d; return v; }
static __device__ __forceinline__ float4 normalize(float4 v) { float d = 1.f / length(v); v.x *= d; v.y *= d; v.z *= d; v.w *= d; return v; }
static __device__ __forceinline__ float clamp(float a, float min, float max) { return a<min ? min : a>max ? max : a; }
static __device__ __forceinline__ float2 clamp(float2 a, float min, float max) { return make_float2(clamp(a.x,min,max),clamp(a.y,min,max)); }
static __device__ __forceinline__ float3 clamp(float3 a, float min, float max) { return make_float3(clamp(a.x,min,max),clamp(a.y,min,max),clamp(a.z,min,max)); }
static __device__ __forceinline__ float4 clamp(float4 a, float min, float max) { return make_float4(clamp(a.x,min,max),clamp(a.y,min,max),clamp(a.z,min,max),clamp(a.w,min,max)); }
static __device__ __forceinline__ void swap(int& a, int& b) { int t = a; a = b; b = t; }
static __device__ __forceinline__ void swap(float& a, float& b) { float t = a; a = b; b = t; }
static __device__ __forceinline__ void swap(float2& a, float2& b) { float2 t = a; a = b; b = t; }
static __device__ __forceinline__ void swap(float3& a, float3& b) { float3 t = a; a = b; b = t; }
static __device__ __forceinline__ void swap(float4& a, float4& b) { float4 t = a; a = b; b = t; }
static __device__ __forceinline__ float cross(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }
static __device__ __forceinline__ float3 cross(float3 a, float3 b) { return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
static __device__ __forceinline__ float lerp(float a, float b, float t) { return a + t * (b - a); }
static __device__ __forceinline__ float2 lerp2(float2 a, float2 b, float t) { return a + t * (b - a); }
static __device__ __forceinline__ float3 lerp3(float3 a, float3 b, float t) { return a + t * (b - a); }
static __device__ __forceinline__ float4 lerp4(float4 a, float4 b, float t) { return a + t * (b - a); }
static __device__ __forceinline__ float bilerp(float a, float b, float c, float d, float2 t) { return lerp(lerp(a, b, t.x), lerp(c, d, t.x), t.y); }
static __device__ __forceinline__ float2 bilerp2(float2 a, float2 b, float2 c, float2 d, float2 t) { return lerp2(lerp2(a, b, t.x), lerp2(c, d, t.x), t.y); }
static __device__ __forceinline__ float3 bilerp3(float3 a, float3 b, float3 c, float3 d, float2 t) { return lerp3(lerp3(a, b, t.x), lerp3(c, d, t.x), t.y); }
static __device__ __forceinline__ float4 bilerp4(float4 a, float4 b, float4 c, float4 d, float2 t) { return lerp4(lerp4(a, b, t.x), lerp4(c, d, t.x), t.y); }
static __device__ __forceinline__ void atomicAdd3(float2* ptr, float2 val) { atomicAdd(&(*ptr).x, val.x); atomicAdd(&(*ptr).y, val.y); }
static __device__ __forceinline__ void atomicAdd3(float3* ptr, float3 val) { atomicAdd(&(*ptr).x, val.x); atomicAdd(&(*ptr).y, val.y); atomicAdd(&(*ptr).z, val.z); }
static __device__ __forceinline__ void atomicAdd4(float4* ptr, float4 val) { atomicAdd(&(*ptr).x, val.x); atomicAdd(&(*ptr).y, val.y); atomicAdd(&(*ptr).z, val.z); atomicAdd(&(*ptr).w, val.w); }
static __device__ __forceinline__ void atomicAdd_xyw(float* ptr, float x, float y, float w) {
	atomicAdd(ptr, x);
	atomicAdd(ptr + 1, y);
	atomicAdd(ptr + 3, w);
}
static __device__ __forceinline__ void AddNaNcheck(float& a, float b) { if (!isfinite(a))a = 0.f; else { float v = a + b; if (isfinite(v))a = v; } };

static __device__ __forceinline__ float getUniform(unsigned int a, unsigned  int b, unsigned int c)
{
	a -= b; a -= c; a ^= (c >> 13);
	b -= c; b -= a; b ^= (a << 8);
	c -= a; c -= b; c ^= (b >> 13);
	a -= b; a -= c; a ^= (c >> 12);
	b -= c; b -= a; b ^= (a << 16);
	c -= a; c -= b; c ^= (b >> 5);
	a -= b; a -= c; a ^= (c >> 3);
	b -= c; b -= a; b ^= (a << 10);
	c -= a; c -= b; c ^= (b >> 15);
	int d = 0x007fffff;
	return float(c & d) / float(d);
}
