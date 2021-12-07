#include "material.h"

void Material::init(MaterialParams& mtr, RasterizeParams& rast, InterpolateParams& pos, InterpolateParams& normal, float* in) {
    mtr.kernel.width = rast.kernel.width;
    mtr.kernel.height = rast.kernel.height;
    mtr.kernel.depth = rast.kernel.depth;
    mtr.kernel.pos = pos.kernel.out;
    mtr.kernel.normal = normal.kernel.out;
    mtr.kernel.rast = rast.kernel.out;
    mtr.kernel.in = in;
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.out, mtr.Size()));
    mtr.block = rast.block;
    mtr.grid = rast.grid;
}

void Material::init(MaterialParams& mtr, float3* eye, int lightNum, float3* lightpos, float3* lightintensity, float3 ambient, float Ka, float Kd, float Ks, float shininess) {
    mtr.kernel.eye = eye;
    mtr.kernel.lightNum = lightNum;
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.lightpos, (size_t)lightNum * sizeof(float3)));
    CUDA_ERROR_CHECK(cudaMemcpy(mtr.kernel.lightpos, lightpos, (size_t)lightNum * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.lightintensity, ((size_t)lightNum + 1) * sizeof(float3)));
    CUDA_ERROR_CHECK(cudaMemcpy(mtr.kernel.lightintensity, &ambient, sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(&mtr.kernel.lightintensity[1], lightintensity, (size_t)lightNum * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.params, (size_t)4 * sizeof(float)));
    float params[4]{ Ka, Kd,  Ks,  shininess };
    CUDA_ERROR_CHECK(cudaMemcpy(mtr.kernel.params, params, (size_t)4 * sizeof(float), cudaMemcpyHostToDevice));
}

__global__ void MaterialForwardKernel(const MaterialKernelParams mtr) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= mtr.width || py >= mtr.height || pz >= mtr.depth)return;
    int pidx = px + mtr.width * (py + mtr.height * pz);

    if (mtr.rast[pidx * 4 + 3] < 1.f) return;

    float3 pos = ((float3*)mtr.pos)[pidx];
    float3 n = normalize(((float3*)mtr.normal)[pidx]);
    float3 v = normalize(*(float3*)&mtr.eye - pos);
    float3 diffuse = make_float3(0.f, 0.f, 0.f);
    float3 specular = make_float3(0.f, 0.f, 0.f);
    for (int i = 0; i < mtr.lightNum; i++) {
        float3 lightpos = mtr.lightpos[i];
        float3 l = normalize(lightpos - pos);
        float ln = dot(l, n);
        float3 r = 2.f * ln * n - l;
        float rv = dot(r, v);
        float3 intensity = mtr.lightintensity[i + 1];
        diffuse += intensity * max(ln, 0.f);
        float powrv = pow(max(rv, 0.f), mtr.params[3]);
        AddNaNcheck(specular.x, intensity.x * powrv);
        AddNaNcheck(specular.y, intensity.y * powrv);
        AddNaNcheck(specular.z, intensity.z * powrv);
    }
    float Ka = mtr.params[0];
    float Kd = mtr.params[1];
    float Ks = mtr.params[2];
    ((float3*)mtr.out)[pidx] = ((float3*)mtr.in)[pidx] * (Ka * mtr.lightintensity[0] + Kd * diffuse + Ks * specular);
}

void Material::forward(MaterialParams& mtr) {
    CUDA_ERROR_CHECK(cudaMemset(mtr.kernel.out, 0, mtr.Size()));
    void* args[] = { &mtr.kernel};
    CUDA_ERROR_CHECK(cudaLaunchKernel(MaterialForwardKernel, mtr.grid, mtr.block, args, 0, NULL));
}

void Material::init(MaterialParams& mtr, float* dLdout) {
    mtr.grad.out = dLdout;
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.grad.in, mtr.Size()));
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.grad.lightpos, (size_t)mtr.kernel.lightNum * sizeof(float3)));
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.grad.lightintensity, ((size_t)mtr.kernel.lightNum + 1) * sizeof(float3)));
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.grad.params, (size_t)4 * sizeof(float)));
}

void Material::clear(MaterialParams& mtr) {
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.out, 0, mtr.Size()));
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.lightpos, 0, (size_t)mtr.kernel.lightNum * sizeof(float3)));
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.lightintensity, 0, ((size_t)mtr.kernel.lightNum + 1) * sizeof(float3)));
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.params, 0, (size_t)4 * sizeof(float)));
}

__global__ void MaterialBackwardKernel(const MaterialKernelParams mtr, const MaterialKernelGradParams grad) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= mtr.width || py >= mtr.height || pz >= mtr.depth)return;
    int pidx = px + mtr.width * (py + mtr.height * pz);

    if (mtr.rast[pidx * 4 + 3] < 1.f) return;

    float3 pos = ((float3*)mtr.pos)[pidx];
    float3 n = ((float3*)mtr.normal)[pidx];
    float3 v = *(float3*)&mtr.eye - pos;
    v *= (1. / sqrt(dot(v, v)));
    float3 diffuse = make_float3(0.f, 0.f, 0.f);
    float3 specular = make_float3(0.f, 0.f, 0.f);
    float dshine = 0.f;
    for (int i = 0; i < mtr.lightNum; i++) {
        float3 light = mtr.lightpos[i];
        //dl/dlight=1
        //dln/dl_=-n_*l_/dot(l,l)/sqrt(dot(l, l))
        float3 l = light - pos;
        l *= (1.f / sqrt(dot(l, l)));
        float ln = dot(l, n);
        //dr/dl_=2*n_*n-1
        //drv/dr=v
        //dspec/drv=shininess*pow(rv,shininess-1)
        float3 r = 2.f * ln * n - l;
        float rv = dot(r, v);
        float3 intensity = mtr.lightintensity[i + 1];
        diffuse += intensity * max(ln, 0.f);
        float powrv = pow(max(rv, 0.f), mtr.params[3]);
        AddNaNcheck(specular.x, intensity.x * powrv);
        AddNaNcheck(specular.y, intensity.y * powrv);
        AddNaNcheck(specular.z, intensity.z * powrv);
        AddNaNcheck(dshine, (intensity.x + intensity.y + intensity.z) * log(max(rv, 0.f)) * powrv);
    }
    float Ka = mtr.params[0];
    float Kd = mtr.params[1];
    float Ks = mtr.params[2];
    float3 dLdout = ((float3*)grad.out)[pidx];
    float3 din = dLdout * ((float3*)mtr.in)[pidx];
    atomicAdd(&grad.params[0], dot(mtr.lightintensity[0], din));
    atomicAdd(&grad.params[1], dot(din,diffuse));
    atomicAdd(&grad.params[2], dot(din,specular));
    atomicAdd(&grad.params[3], Ks * dshine);
    ((float3*)grad.in)[pidx] = dLdout * (Ka * mtr.lightintensity[0] + Kd * diffuse + Ks * specular);
}

void Material::backward(MaterialParams& mtr) {
    void* args[] = { &mtr.kernel, &mtr.grad };
    CUDA_ERROR_CHECK(cudaLaunchKernel(MaterialBackwardKernel, mtr.grid, mtr.block, args, 0, NULL));
}