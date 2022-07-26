#include "material.h"

void Material::init(MaterialParams& mtr, RasterizeParams& rast, ProjectParams& pos, ProjectParams& normal, Attribute* texel, int channel, float* in) {
    if (pos.kernel.dimention != 3)ERROR_STRING(dimention is not 3);
    if (normal.kernel.dimention != 3)ERROR_STRING(dimention is not 3);
    mtr.kernel.width = rast.kernel.width;
    mtr.kernel.height = rast.kernel.height;
    mtr.kernel.depth = rast.kernel.depth;
    mtr.kernel.channel = channel;
    mtr.kernel.pos = pos.kernel.out;
    mtr.kernel.normal = normal.kernel.out;
    mtr.kernel.texel = texel->vbo;
    mtr.kernel.posidx = pos.vao;
    mtr.kernel.normalidx = normal.vao;
    mtr.kernel.texelidx = texel->vao;
    mtr.kernel.rast = rast.kernel.out;
    mtr.kernel.in = in;
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.out, mtr.Size()));
}

void Material::init(MaterialParams& mtr, float3 eye, Buffer& point, Buffer& intensity) {
    mtr.kernel.eye = eye;
    mtr.kernel.lightNum = point.num;
    mtr.kernel.point = point.buffer;
    mtr.kernel.intensity = intensity.buffer;
}

void Material::init(MaterialParams& mtr, Buffer& params) {
    mtr.kernel.params = params.buffer;;
}

void Material::init(MaterialGradParams& mtr, RasterizeParams& rast, ProjectParams& pos, ProjectParams& normal, Attribute* texel, int channel, float* in, float* grad) {
    init((MaterialParams&)mtr, rast, pos, normal, texel, channel, in);
    mtr.grad.in = grad;
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.grad.out, mtr.Size()));
}

void Material::init(MaterialGradParams& mtr, float3 eye, BufferGrad& point, BufferGrad& intensity) {
    Material::init((MaterialParams&)mtr, eye, point, intensity); 
    mtr.grad.point = point.grad; 
    mtr.grad.intensity = intensity.grad;
}

void Material::init(MaterialGradParams& mtr, BufferGrad& params) {
    Material::init((MaterialParams&)mtr, params); 
    mtr.grad.params = params.grad;
}



__global__ void PhongMaterialForwardKernel(const MaterialKernelParams mtr) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= mtr.width || py >= mtr.height || pz >= mtr.depth)return;
    int pidx = px + mtr.width * (py + mtr.height * pz);

    float4 r = ((float4*)mtr.rast)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0) return;

    float3 p2 = ((float3*)mtr.pos)[mtr.posidx[idx * 3 + 2]];
    float3 p0 = ((float3*)mtr.pos)[mtr.posidx[idx * 3]] - p2;
    float3 p1 = ((float3*)mtr.pos)[mtr.posidx[idx * 3 + 1]] - p2;
    float3 n2 = ((float3*)mtr.normal)[mtr.normalidx[idx * 3 + 2]];
    float3 n0 = ((float3*)mtr.normal)[mtr.normalidx[idx * 3]] - n2;
    float3 n1 = ((float3*)mtr.normal)[mtr.normalidx[idx * 3 + 1]] - n2;

    float3 pos = p0 * r.x + p1 * r.y + p2;
    float3 n = normalize(n0 * r.x + n1 * r.y + n2);
    float3 v = normalize(*(float3*)&mtr.eye - pos);

    float Ka = mtr.params[0];
    float Kd = mtr.params[1];
    float Ks = mtr.params[2];
    float shininess = mtr.params[3];

    for (int k = 0; k < mtr.channel; k++) {
        mtr.out[pidx * mtr.channel + k] += mtr.in[pidx * mtr.channel + k] * Ka;
    }
    for (int i = 0; i < mtr.lightNum; i++) {
        float3 l = ((float3*)mtr.point)[i] - pos;
        float il2 = 1.f / dot(l, l);
        l *= sqrt(il2);
        float ln = dot(l, n);
        float3 r = 2.f * ln * n - l;
        float rv = dot(r, v);
        float powrv = pow(max(rv, 0.f), shininess);
        ln = max(ln, 0.f);
        for (int k = 0; k < mtr.channel; k++) {
            float intensity = mtr.intensity[i * mtr.channel + k] * il2;
            float diffuse = intensity * ln;
            float specular = intensity * powrv;
            mtr.out[pidx * mtr.channel + k] += mtr.in[pidx * mtr.channel + k] * Kd * diffuse + Ks * specular;
        }
    }
}

void PhongMaterial::forward(MaterialParams& mtr) {
    CUDA_ERROR_CHECK(cudaMemset(mtr.kernel.out, 0, mtr.Size()));
    dim3 block = getBlock(mtr.kernel.width, mtr.kernel.height);
    dim3 grid = getGrid(block, mtr.kernel.width, mtr.kernel.height, mtr.kernel.depth);
    void* args[] = { &mtr.kernel };
    CUDA_ERROR_CHECK(cudaLaunchKernel(PhongMaterialForwardKernel, grid, block, args, 0, NULL));
}


void PhongMaterial::forward(MaterialGradParams& mtr) {
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.out, 0, mtr.Size()));
    forward((MaterialParams&)mtr);
}

__global__ void PhongMaterialBackwardKernel(const MaterialKernelParams mtr, const MaterialKernelGradParams grad) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= mtr.width || py >= mtr.height || pz >= mtr.depth)return;
    int pidx = px + mtr.width * (py + mtr.height * pz);

    float4 r = ((float4*)mtr.rast)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0) return;

    float3 p2 = ((float3*)mtr.pos)[mtr.posidx[idx * 3 + 2]];
    float3 p0 = ((float3*)mtr.pos)[mtr.posidx[idx * 3]] - p2;
    float3 p1 = ((float3*)mtr.pos)[mtr.posidx[idx * 3 + 1]] - p2;
    float3 n2 = ((float3*)mtr.normal)[mtr.normalidx[idx * 3 + 2]];
    float3 n0 = ((float3*)mtr.normal)[mtr.normalidx[idx * 3]] - n2;
    float3 n1 = ((float3*)mtr.normal)[mtr.normalidx[idx * 3 + 1]] - n2;

    float3 pos = p0 * r.x + p1 * r.y + p2;
    float3 n = normalize(n0 * r.x + n1 * r.y + n2);
    float3 v = normalize(mtr.eye - pos);

    float Ka = mtr.params[0];
    float Kd = mtr.params[1];
    float Ks = mtr.params[2];

    for (int k = 0; k < mtr.channel; k++) {
        float dLdout = grad.out[pidx * mtr.channel + k];
        float din = mtr.in[pidx * mtr.channel + k] * dLdout;
        if(grad.params!=nullptr)atomicAdd(&grad.params[0], din);
        if (grad.in != nullptr)grad.in[pidx * mtr.channel + k] += Ka * dLdout;
    }
    for (int i = 0; i < mtr.lightNum; i++) {
        float3 l = ((float3*)mtr.point)[i] - pos;
        float il2 = 1.f / dot(l, l);
        l *= sqrt(il2);
        float ln = dot(l, n);
        float3 r = 2.f * ln * n - l;
        float rv = dot(r, v);
        float dkd = 0.f;
        float dks = 0.f;
        float powrv = pow(max(rv, 0.f), mtr.params[3]);
        for (int k = 0; k < mtr.channel; k++) {
            float dLdout = grad.out[pidx * mtr.channel + k];
            float din = dLdout * mtr.intensity[i * mtr.channel + k] * il2;
            float dln = ln * din;
            if (grad.in != nullptr)grad.in[pidx * mtr.channel + k] += Kd * dln;
            dkd += dln * mtr.in[pidx * mtr.channel + k];
            dks += din;
        }
        if (grad.params != nullptr) {
            atomicAdd(&grad.params[1], dkd);
            dks *= powrv;
            atomicAdd(&grad.params[2], dks);
            atomicAdd(&grad.params[3], Ks * dks * log(max(rv, 1e-3)));
        }
    }
}

void PhongMaterial::backward(MaterialGradParams& mtr) {
    dim3 block = getBlock(mtr.kernel.width, mtr.kernel.height);
    dim3 grid = getGrid(block, mtr.kernel.width, mtr.kernel.height, mtr.kernel.depth);
    void* args[] = { &mtr.kernel, &mtr.grad };
    CUDA_ERROR_CHECK(cudaLaunchKernel(PhongMaterialBackwardKernel, grid, block, args, 0, NULL));
}