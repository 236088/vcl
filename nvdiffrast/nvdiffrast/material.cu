#include "material.h"

void Material::init(MaterialParams& mtr, RasterizeParams& rast, InterpolateParams& pos, InterpolateParams& normal, int channel, float* in) {
    mtr.kernel.width = rast.kernel.width;
    mtr.kernel.height = rast.kernel.height;
    mtr.kernel.depth = rast.kernel.depth;
    mtr.kernel.channel = channel;
    mtr.kernel.pos = pos.kernel.out;
    mtr.kernel.normal = normal.kernel.out;
    mtr.kernel.rast = rast.kernel.out;
    mtr.kernel.in = in;
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.out, mtr.Size()));
    mtr.block = rast.block;
    mtr.grid = rast.grid;
}

void Material::init(MaterialParams& mtr, float3* eye, int lightNum, float3* direction, float3* lightintensity) {
    mtr.kernel.eye = eye;
    mtr.kernel.lightNum = lightNum;
    for (int i = 0; i < lightNum; i++) {
        float r = sqrt(direction->x * direction->x + direction->y * direction->y + direction->z * direction->z);
        direction->x /= r;
        direction->y /= r;
        direction->z /= r;
    }
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.direction, (size_t)lightNum * sizeof(float3)));
    CUDA_ERROR_CHECK(cudaMemcpy(mtr.kernel.direction, direction, (size_t)lightNum * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.lightintensity, (size_t)lightNum * mtr.kernel.channel * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(mtr.kernel.lightintensity, lightintensity, (size_t)lightNum * mtr.kernel.channel * sizeof(float), cudaMemcpyHostToDevice));
}

void Material::init(MaterialGradParams& mtr, RasterizeParams& rast, InterpolateParams& pos, InterpolateParams& normal, int channel, float* in, float* grad) {
    init((MaterialParams&)mtr, rast, pos, normal, channel, in);
    mtr.grad.in = grad;
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.grad.out, mtr.Size()));
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.grad.direction, (size_t)mtr.kernel.lightNum * sizeof(float3)));
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.grad.lightintensity, (size_t)mtr.kernel.lightNum * mtr.kernel.channel * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.grad.params, (size_t)4 * sizeof(float)));
}



void PhongMaterial::init(MaterialParams& mtr, float3* eye, int lightNum, float3* direction, float3* lightintensity, float Ka, float Kd, float Ks, float shininess) {
    Material::init(mtr, eye, lightNum, direction, lightintensity);
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.params, (size_t)4 * sizeof(float)));
    float params[4]{ Ka, Kd,  Ks,  shininess };
    CUDA_ERROR_CHECK(cudaMemcpy(mtr.kernel.params, params, (size_t)4 * sizeof(float), cudaMemcpyHostToDevice));
}

__global__ void PhongMaterialForwardKernel(const MaterialKernelParams mtr) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= mtr.width || py >= mtr.height || pz >= mtr.depth)return;
    int pidx = px + mtr.width * (py + mtr.height * pz);

    if (mtr.rast[pidx * 4 + 3] < 1.f) return;

    float Ka = mtr.params[0];
    float Kd = mtr.params[1];
    float Ks = mtr.params[2];
    float shininess = mtr.params[3];

    float3 pos = ((float3*)mtr.pos)[pidx];
    float3 n = normalize(((float3*)mtr.normal)[pidx]);
    float3 v = normalize(*(float3*)&mtr.eye - pos);
    for (int i = 0; i < mtr.lightNum; i++) {
        float3 direction = mtr.direction[i];
        float3 l = normalize(direction - pos);
        float ln = dot(l, n);
        float3 r = 2.f * ln * n - l;
        float rv = dot(r, v);
        float powrv = pow(max(rv, 0.f), shininess);
        for (int k = 0; k < mtr.channel; k++) {
            float intensity = mtr.lightintensity[i * mtr.channel + k];
            float diffuse = intensity * max(ln, 0.f);
            float specular = intensity * powrv;
            mtr.out[pidx * mtr.channel + k] += mtr.in[pidx * mtr.channel + k] * (Ka + Kd * diffuse + Ks * specular);
        }
    }
}

void PhongMaterial::forward(MaterialParams& mtr) {
    CUDA_ERROR_CHECK(cudaMemset(mtr.kernel.out, 0, mtr.Size()));
    void* args[] = { &mtr.kernel };
    CUDA_ERROR_CHECK(cudaLaunchKernel(PhongMaterialForwardKernel, mtr.grid, mtr.block, args, 0, NULL));
}

void PhongMaterial::forward(MaterialGradParams& mtr) {
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.out, 0, mtr.Size()));
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.direction, 0, (size_t)mtr.kernel.lightNum * sizeof(float3)));
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.lightintensity, 0, (size_t)mtr.kernel.lightNum * mtr.kernel.channel * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.params, 0, (size_t)4 * sizeof(float)));
    forward((MaterialParams&)mtr);
}

__global__ void PhongMaterialBackwardKernel(const MaterialKernelParams mtr, const MaterialKernelGradParams grad) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= mtr.width || py >= mtr.height || pz >= mtr.depth)return;
    int pidx = px + mtr.width * (py + mtr.height * pz);

    if (mtr.rast[pidx * 4 + 3] < 1.f) return;

    float Ka = mtr.params[0];
    float Kd = mtr.params[1];
    float Ks = mtr.params[2];

    float3 pos = ((float3*)mtr.pos)[pidx];
    float3 n = normalize(((float3*)mtr.normal)[pidx]);
    float3 v = normalize(*(float3*)&mtr.eye - pos);

    for (int i = 0; i < mtr.lightNum; i++) {
        float3 direction = mtr.direction[i];
        float3 l = normalize(direction - pos);
        float ln = dot(l, n);
        float3 r = 2.f * ln * n - l;
        float rv = dot(r, v);

        float powrv = pow(max(rv, 0.f), mtr.params[3]);
        for (int k = 0; k < mtr.channel; k++) {
            float dLdout = grad.out[pidx * mtr.channel + k];
            float din = dLdout * mtr.in[pidx * mtr.channel + k];
            float intensity = mtr.lightintensity[i * mtr.channel + k];
            float diffuse = intensity * max(ln, 0.f);
            float specular = intensity * powrv;
            float dshine = intensity * log(max(rv, 1e-3)) * powrv;
            grad.in[pidx * mtr.channel + k] = dLdout * (Ka + Kd * diffuse + Ks * specular);
            atomicAdd(&grad.params[0], din);
            atomicAdd(&grad.params[1], din * diffuse);
            atomicAdd(&grad.params[2], din * specular);
            atomicAdd(&grad.params[3], Ks * dshine);
        }
    }
}

void PhongMaterial::backward(MaterialGradParams& mtr) {
    void* args[] = { &mtr.kernel, &mtr.grad };
    CUDA_ERROR_CHECK(cudaLaunchKernel(PhongMaterialBackwardKernel, mtr.grid, mtr.block, args, 0, NULL));
}


void PBRMaterial::init(MaterialParams& mtr, float3* eye, int lightNum, float3* direction, float3* lightintensity, float roughness, float ior) {
    Material::init(mtr, eye, lightNum, direction, lightintensity);
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.params, (size_t)2 * sizeof(float)));
    float params[2]{roughness, ior };
    CUDA_ERROR_CHECK(cudaMemcpy(mtr.kernel.params, params, (size_t)2 * sizeof(float), cudaMemcpyHostToDevice));
}

__global__ void PBRMaterialForwardKernel(const MaterialKernelParams mtr) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= mtr.width || py >= mtr.height || pz >= mtr.depth)return;
    int pidx = px + mtr.width * (py + mtr.height * pz);

    if (mtr.rast[pidx * 4 + 3] < 1.f) return;

    float m2 = mtr.params[0] * mtr.params[0];
    float n2 = mtr.params[1] * mtr.params[1] - 1.f;

    float3 pos = ((float3*)mtr.pos)[pidx];
    float3 n = normalize(((float3*)mtr.normal)[pidx]);
    float3 v = normalize(*(float3*)&mtr.eye - pos);
    float nv = max(dot(n, v), 0.f);
    for (int i = 0; i < mtr.lightNum; i++) {
        float3 l = -mtr.direction[i];
        float3 h = normalize(l + v);
        float nl = max(dot(n, l), 0.f);
        float nh = max(dot(n, h), 0.f);
        float lh = max(dot(l, h), 0.f);
        float D = (nh * nh * (m2 - 1.f) + 1.f);
        D = m2 / (D * D * 3.14159265f);
        float Vl = 1.f / (nl + sqrt(m2 + nl * nl * (1.f - m2)));
        float Vv = 1.f / (nv + sqrt(m2 + nv * nv * (1.f - m2)));
        float g = sqrt(n2 + lh * lh);
        float g_nlh = g - lh;
        float g_plh = g + lh;
        float F = (g_nlh / g_plh);
        F *= F * .5f;
        float F1 = (lh * g_plh - 1.f) / (lh * g_nlh + 1.f);
        F *= (1.f + F1 * F1);
        float specular = nl * D * Vl * Vv * F;
        float diffuse = nl / 3.14159265f;
        for (int k = 0; k < mtr.channel; k++) {
            float intensity = mtr.lightintensity[i * mtr.channel + k];
            mtr.out[pidx * mtr.channel + k] += intensity * (mtr.in[pidx * mtr.channel + k] * diffuse + specular);
        }
    }       
    //((float3*)mtr.out)[pidx] = clamp(((float3*)mtr.out)[pidx], 0.f, 1.f);
}

void PBRMaterial::forward(MaterialParams& mtr) {
    CUDA_ERROR_CHECK(cudaMemset(mtr.kernel.out, 0, mtr.Size()));
    void* args[] = { &mtr.kernel };
    CUDA_ERROR_CHECK(cudaLaunchKernel(PBRMaterialForwardKernel, mtr.grid, mtr.block, args, 0, NULL));
}

void PBRMaterial::forward(MaterialGradParams& mtr) {
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.out, 0, mtr.Size()));
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.direction, 0, (size_t)mtr.kernel.lightNum * sizeof(float3)));
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.lightintensity, 0, (size_t)mtr.kernel.lightNum * mtr.kernel.channel * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.params, 0, (size_t)2 * sizeof(float)));
    forward((MaterialParams&)mtr);
}

__global__ void PBRMaterialBackwardKernel(const MaterialKernelParams mtr, const MaterialKernelGradParams grad) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= mtr.width || py >= mtr.height || pz >= mtr.depth)return;
    int pidx = px + mtr.width * (py + mtr.height * pz);

    if (mtr.rast[pidx * 4 + 3] < 1.f) return;

    float Ka = mtr.params[0];
    float Kd = mtr.params[1];
    float Ks = mtr.params[2];

    float3 pos = ((float3*)mtr.pos)[pidx];
    float3 n = normalize(((float3*)mtr.normal)[pidx]);
    float3 v = normalize(*(float3*)&mtr.eye - pos);

    for (int i = 0; i < mtr.lightNum; i++) {
        float3 direction = mtr.direction[i];
        float3 l = normalize(direction - pos);
        float ln = dot(l, n);
        float3 r = 2.f * ln * n - l;
        float rv = dot(r, v);

        float powrv = pow(max(rv, 0.f), mtr.params[3]);
        for (int k = 0; k < mtr.channel; k++) {
            float dLdout = grad.out[pidx * mtr.channel + k];
            float din = dLdout * mtr.in[pidx * mtr.channel + k];
            float intensity = mtr.lightintensity[i * mtr.channel + k];
            float diffuse = intensity * max(ln, 0.f);
            float specular = intensity * powrv;
            float dshine = intensity * log(max(rv, 1e-3)) * powrv;
            grad.in[pidx * mtr.channel + k] = dLdout * (Ka + Kd * diffuse + Ks * specular);
            atomicAdd(&grad.params[0], din);
            atomicAdd(&grad.params[1], din * diffuse);
            atomicAdd(&grad.params[2], din * specular);
            atomicAdd(&grad.params[3], Ks * dshine);
        }
    }
}

void PBRMaterial::backward(MaterialGradParams& mtr) {
    void* args[] = { &mtr.kernel, &mtr.grad };
    CUDA_ERROR_CHECK(cudaLaunchKernel(PBRMaterialBackwardKernel, mtr.grid, mtr.block, args, 0, NULL));
}