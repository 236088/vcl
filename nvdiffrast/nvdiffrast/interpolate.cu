#include "interpolate.h"

void Interpolate::init(InterpolateParams& intr, RasterizeParams& rast, Attribute& attr) {
    intr.kernel.width = rast.kernel.width;
    intr.kernel.height = rast.kernel.height;
    intr.kernel.depth = rast.kernel.depth;
    intr.kernel.enableDA = rast.kernel.enableDB;
    intr.kernel.rast = rast.kernel.out;
    intr.attrNum = attr.vboNum;
    intr.idxNum = attr.vaoNum;
    intr.kernel.dimention = attr.dimention;
    intr.kernel.attr = attr.vbo;
    intr.kernel.idx = attr.vao;

    CUDA_ERROR_CHECK(cudaMalloc(&intr.kernel.out, intr.Size()));
    if (intr.kernel.enableDA) {
        intr.kernel.rastDB = rast.kernel.outDB;
        CUDA_ERROR_CHECK(cudaMalloc(&intr.kernel.outDA, intr.Size() * 2));
    }
    intr.block = rast.block;
    intr.grid = rast.grid;
}

__global__ void InterplateForwardKernel(const InterpolateKernelParams intr) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= intr.width || py >= intr.height || pz >= intr.depth)return;
    int pidx = px + intr.width * (py + intr.height * pz);

    float4 r = ((float4*)intr.rast)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0) return;

    unsigned int idx0 = intr.idx[idx * 3];
    unsigned int idx1 = intr.idx[idx * 3 + 1];
    unsigned int idx2 = intr.idx[idx * 3 + 2];
    for (int i = 0; i < intr.dimention; i++) {
        float a0 = intr.attr[idx0 * intr.dimention + i];
        float a1 = intr.attr[idx1 * intr.dimention + i];
        float a2 = intr.attr[idx2 * intr.dimention + i];
        intr.out[pidx * intr.dimention + i] = a0 * r.x + a1 * r.y + a2 * (1.0 - r.x - r.y);

        if (intr.enableDA) {
            float dadu = a0 - a2;
            float dadv = a1 - a2;
            float4 db = ((float4*)intr.rastDB)[pidx];

            ((float2*)intr.outDA)[pidx * intr.dimention + i] =
                make_float2(dadu * db.x + dadv * db.z, dadu * db.y + dadv * db.w);
        }
    }
}

void Interpolate::forward(InterpolateParams& intr) {
    CUDA_ERROR_CHECK(cudaMemset(intr.kernel.out, 0, intr.Size()));
    if (intr.kernel.enableDA) {
        CUDA_ERROR_CHECK(cudaMemset(intr.kernel.outDA, 0, intr.Size() * 2));
    }
    void* args[] = { &intr.kernel };
    CUDA_ERROR_CHECK(cudaLaunchKernel(InterplateForwardKernel, intr.grid, intr.block, args, 0, NULL));
}

void Interpolate::forward(InterpolateGradParams& intr) {
    CUDA_ERROR_CHECK(cudaMemset(intr.grad.out, 0, intr.Size()));
    if (intr.kernel.enableDA) CUDA_ERROR_CHECK(cudaMemset(intr.grad.outDA, 0, intr.Size() * 2));
    forward((InterpolateParams&)intr);
}

void Interpolate::init(InterpolateGradParams& intr, RasterizeGradParams& rast, AttributeGrad& attr) {
    init((InterpolateParams&)intr, rast, attr);
    CUDA_ERROR_CHECK(cudaMalloc(&intr.grad.out, intr.Size()));
    if (intr.kernel.enableDA) CUDA_ERROR_CHECK(cudaMalloc(&intr.grad.outDA, intr.Size() * 2));
    intr.grad.attr = attr.grad;
    intr.grad.rast = rast.grad.out;
}

void Interpolate::init(InterpolateGradParams& intr, RasterizeGradParams& rast, Attribute& attr) {
    init((InterpolateParams&)intr, rast, attr);
    CUDA_ERROR_CHECK(cudaMalloc(&intr.grad.out, intr.Size()));
    if (intr.kernel.enableDA) CUDA_ERROR_CHECK(cudaMalloc(&intr.grad.outDA, intr.Size() * 2));
    intr.grad.attr = nullptr;
    intr.grad.rast = rast.grad.out;
}

void Interpolate::init(InterpolateGradParams& intr, RasterizeParams& rast, AttributeGrad& attr) {
    init((InterpolateParams&)intr, rast, attr);
    CUDA_ERROR_CHECK(cudaMalloc(&intr.grad.out, intr.Size()));
    if (intr.kernel.enableDA) CUDA_ERROR_CHECK(cudaMalloc(&intr.grad.outDA, intr.Size() * 2));
    intr.grad.attr = attr.grad;
    intr.grad.rast = nullptr;
}

// a = u * (a0 - a2) + v * (a1 - a2) + a2
// dL/da0 = dL/da * da/da0 = dL/da * u
// dL/da1 = dL/da * da/da1 = dL/da * v
// dL/da2 = dL/da * da/da2 = dL/da * (1 - u - v)
//
// dL/du = dot(dL/da, da/du) = dot(dL/da, (a0 - a2))
// dL/dv = dot(dL/da, da/dv) = dot(dL/da, (a1 - a2))
//
//
// da/dx = da/du * du/dx + da/dv * dv/dx = (a0 - a2) * du/dx + (a1 - a2) * dv/dx
// da/dy = da/du * du/dy + da/dv * dv/dy = (a0 - a2) * du/dy + (a1 - a2) * dv/dy
//
// dL/d(du/dx) = dot(dL/d(da/dx), da/du) = dot(dL/d(da/dx), (a0 - a2))
// dL/d(du/dy) = dot(dL/d(da/dy), da/du) = dot(dL/d(da/dy), (a0 - a2))
// dL/d(dv/dx) = dot(dL/d(da/dx), da/dv) = dot(dL/d(da/dx), (a1 - a2))
// dL/d(dv/dy) = dot(dL/d(da/dy), da/dv) = dot(dL/d(da/dy), (a1 - a2))
//
// dL/da0 = dL/d(da/dx) * d(da/dx)/da0 + dL/d(da/dy) * d(da/dy)/da0 = dL/d(da/dx) * du/dx + dL/d(da/dy) * du/dy
// dL/da1 = dL/d(da/dx) * d(da/dx)/da1 + dL/d(da/dy) * d(da/dy)/da1 = dL/d(da/dx) * dv/dx + dL/d(da/dy) * dv/dy
// dL/da2 = dL/d(da/dx) * d(da/dx)/da2 + dL/d(da/dy) * d(da/dy)/da2
//        = -dL/d(da/dx) * du/dx - dL/d(da/dy) * du/dy - dL/d(da/dx) * dv/dx - dL/d(da/dy) * dv/dy = -dL/da0 - dL/da1

__global__ void InterpolateBackwardKernel(const InterpolateKernelParams intr, const InterpolateKernelGradParams grad) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= intr.width || py >= intr.height || pz >= intr.depth)return;
    int pidx = px + intr.width * (py + intr.height * pz);

    float4 r = ((float4*)intr.rast)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0) return;

    unsigned int idx0 = intr.idx[idx * 3];
    unsigned int idx1 = intr.idx[idx * 3 + 1];
    unsigned int idx2 = intr.idx[idx * 3 + 2];
    float gu = 0.f;
    float gv = 0.f;
    for (int i = 0; i < intr.dimention; i++) {
        float dLdout = grad.out[pidx * intr.dimention + i];
        if (grad.attr != nullptr) {
            atomicAdd(&grad.attr[idx0 * intr.dimention + i], dLdout * r.x);
            atomicAdd(&grad.attr[idx1 * intr.dimention + i], dLdout * r.y);
            atomicAdd(&grad.attr[idx2 * intr.dimention + i], dLdout * (1.0 - r.x - r.y));
        }
        gu += dLdout * (intr.attr[idx0 * intr.dimention + i] - intr.attr[idx2 * intr.dimention + i]);
        gv += dLdout * (intr.attr[idx1 * intr.dimention + i] - intr.attr[idx2 * intr.dimention + i]);
    }
    if (grad.rast != nullptr) {
        ((float2*)grad.rast)[pidx] = make_float2(gu, gv);
    }
}

void Interpolate::backward(InterpolateGradParams& intr) {
    void* args[] = { &intr.kernel,&intr.grad};
    CUDA_ERROR_CHECK(cudaLaunchKernel(InterpolateBackwardKernel, intr.grid, intr.block, args, 0, NULL));
}