#include "material.h"

void Material::init(MaterialParams& mtr, RasterizeParams& rast, ProjectParams& pos, ProjectParams& normal, Attribute* texel, int channel, float* diffusemap, float* roughnessmap, float* normalmap, float* heightmap) {
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
    mtr.kernel.diffusemap = diffusemap;
    mtr.kernel.roughnessmap = roughnessmap;
    mtr.kernel.normalmap = normalmap;
    mtr.kernel.heightmap = heightmap;
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.out, mtr.Size()));
    mtr.block = rast.block;
    mtr.grid = rast.grid;
}

void Material::init(MaterialParams& mtr, float3* eye, int lightNum, float3* point, float3* lightintensity) {
    mtr.kernel.eye = eye;
    mtr.kernel.lightNum = lightNum;

    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.point, (size_t)lightNum * sizeof(float3)));
    CUDA_ERROR_CHECK(cudaMemcpy(mtr.kernel.point, point, (size_t)lightNum * sizeof(float3), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.lightintensity, (size_t)lightNum * mtr.kernel.channel * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(mtr.kernel.lightintensity, lightintensity, (size_t)lightNum * mtr.kernel.channel * sizeof(float), cudaMemcpyHostToDevice));
}

void Material::init(MaterialGradParams& mtr, RasterizeParams& rast, ProjectParams& pos, ProjectParams& normal, Attribute& texel, int channel, float* in, float* grad) {
    init((MaterialParams&)mtr, rast, pos, normal, nullptr, channel, in, nullptr, nullptr, nullptr);
    mtr.grad.diffusemap = grad;
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.grad.out, mtr.Size()));
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.grad.point, (size_t)mtr.kernel.lightNum * sizeof(float3)));
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.grad.lightintensity, (size_t)mtr.kernel.lightNum * mtr.kernel.channel * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.grad.params, (size_t)4 * sizeof(float)));
}



void PhongMaterial::init(MaterialParams& mtr, float3* eye, int lightNum, float3* point, float3* lightintensity, float Ka, float Kd, float Ks, float shininess) {
    Material::init(mtr, eye, lightNum, point, lightintensity);
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

    for (int i = 0; i < mtr.lightNum; i++) {
        float3 point = mtr.point[i];
        float3 l = normalize(point - pos);
        float ln = dot(l, n);
        float3 r = 2.f * ln * n - l;
        float rv = dot(r, v);
        float powrv = pow(max(rv, 0.f), shininess);
        for (int k = 0; k < mtr.channel; k++) {
            float intensity = mtr.lightintensity[i * mtr.channel + k];
            float diffuse = intensity * max(ln, 0.f);
            float specular = intensity * powrv;
            mtr.out[pidx * mtr.channel + k] += mtr.diffusemap[pidx * mtr.channel + k] * (Ka + Kd * diffuse + Ks * specular);
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
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.point, 0, (size_t)mtr.kernel.lightNum * sizeof(float3)));
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

    for (int i = 0; i < mtr.lightNum; i++) {
        float3 point = mtr.point[i];
        float3 l = normalize(point - pos);
        float ln = dot(l, n);
        float3 r = 2.f * ln * n - l;
        float rv = dot(r, v);

        float powrv = pow(max(rv, 0.f), mtr.params[3]);
        for (int k = 0; k < mtr.channel; k++) {
            float dLdout = grad.out[pidx * mtr.channel + k];
            float din = dLdout * mtr.diffusemap[pidx * mtr.channel + k];
            float intensity = mtr.lightintensity[i * mtr.channel + k];
            float diffuse = intensity * max(ln, 0.f);
            float specular = intensity * powrv;
            float dshine = intensity * log(max(rv, 1e-3)) * powrv;
            grad.diffusemap[pidx * mtr.channel + k] = dLdout * (Ka + Kd * diffuse + Ks * specular);
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



void PBRMaterial::init(MaterialParams& mtr, float3* eye, int lightNum, float3* point, float3* lightintensity, float ior) {
    Material::init(mtr, eye, lightNum, point, lightintensity);
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.params, sizeof(float)));
    float params[1]{ ior };
    CUDA_ERROR_CHECK(cudaMemcpy(mtr.kernel.params, params, sizeof(float), cudaMemcpyHostToDevice));
}

__global__ void PBRMaterialForwardKernel(const MaterialKernelParams mtr) {
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
    float2 uv2 = ((float2*)mtr.texel)[mtr.texelidx[idx * 3 + 2]];
    float2 uv0 = ((float2*)mtr.texel)[mtr.texelidx[idx * 3]] - uv2;
    float2 uv1 = ((float2*)mtr.texel)[mtr.texelidx[idx * 3 + 1]] - uv2;

    float3 pos = p0 * r.x + p1 * r.y + p2;
    float3 normal = normalize(n0 * r.x + n1 * r.y + n2);
    float3 tangent = normalize(p0 * uv1.y - p1 * uv0.y);
    float3 bitangent = normalize(p1 * uv0.x - p0 * uv1.x);

    float3 n = ((float3*)mtr.normalmap)[pidx];
    n = 2.f * n - 1.f;
    n = normalize(tangent * n.x + bitangent * n.y + normal * n.z);
    float3 v = normalize(*(float3*)&mtr.eye - pos);

    float m_2 = mtr.roughnessmap[pidx] * mtr.roughnessmap[pidx];
    float n_2 = mtr.params[0] * mtr.params[0] - 1.f;
    float disp = mtr.heightmap[pidx];

    float nv = max(dot(n, v), 0.f);
    for (int i = 0; i < mtr.lightNum; i++) {
        float3 l = normalize(mtr.point[i] - pos);
        float3 h = normalize(l + v);
        float nl = max(dot(n, l), 0.f);
        float nh = max(dot(n, h), 0.f);
        float lh = max(dot(l, h), 0.f);
        float D = (nh * nh * (m_2 - 1.f) + 1.f);
        D = m_2 / (D * D * 3.14159265f);
        float Vl = 1.f / (nl + sqrt(m_2 + nl * nl * (1.f - m_2)));
        float Vv = 1.f / (nv + sqrt(m_2 + nv * nv * (1.f - m_2)));
        float g = sqrt(n_2 + lh * lh);
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
            mtr.out[pidx * mtr.channel + k] += intensity * (mtr.diffusemap[pidx * mtr.channel + k] * diffuse + specular);
        }
    }       
    ((float3*)mtr.out)[pidx] = clamp(((float3*)mtr.out)[pidx], 0.f, 1.f);
}

void PBRMaterial::forward(MaterialParams& mtr) {
    CUDA_ERROR_CHECK(cudaMemset(mtr.kernel.out, 0, mtr.Size()));
    void* args[] = { &mtr.kernel };
    CUDA_ERROR_CHECK(cudaLaunchKernel(PBRMaterialForwardKernel, mtr.grid, mtr.block, args, 0, NULL));
}

void PBRMaterial::forward(MaterialGradParams& mtr) {
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.out, 0, mtr.Size()));
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.point, 0, (size_t)mtr.kernel.lightNum * sizeof(float3)));
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.lightintensity, 0, (size_t)mtr.kernel.lightNum * mtr.kernel.channel * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.params, 0, sizeof(float)));
    forward((MaterialParams&)mtr);
}

__global__ void PBRMaterialBackwardKernel(const MaterialKernelParams mtr, const MaterialKernelGradParams grad) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= mtr.width || py >= mtr.height || pz >= mtr.depth)return;
    int pidx = px + mtr.width * (py + mtr.height * pz);

    float4 r = ((float4*)mtr.rast)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0) return;

    float3 p0 = ((float3*)mtr.pos)[mtr.posidx[idx * 3]];
    float3 p1 = ((float3*)mtr.pos)[mtr.posidx[idx * 3 + 1]];
    float3 p2 = ((float3*)mtr.pos)[mtr.posidx[idx * 3 + 2]];
    float3 n0 = ((float3*)mtr.normal)[mtr.normalidx[idx * 3]];
    float3 n1 = ((float3*)mtr.normal)[mtr.normalidx[idx * 3 + 1]];
    float3 n2 = ((float3*)mtr.normal)[mtr.normalidx[idx * 3 + 2]];

    float3 pos = p0 * r.x + p1 * r.y + p2 * (1.0 - r.x - r.y);
    float3 n = normalize(n0 * r.x + n1 * r.y + n2 * (1.0 - r.x - r.y));
    float3 v = normalize(*(float3*)&mtr.eye - pos);

    float m_2 = mtr.roughnessmap[pidx] * mtr.roughnessmap[pidx];
    float n_2 = mtr.params[0] * mtr.params[0] - 1.f;

    float nv = max(dot(n, v), 0.f);
    for (int i = 0; i < mtr.lightNum; i++) {
        float3 l = -mtr.point[i];
        float3 h = normalize(l + v);
        float nl = max(dot(n, l), 0.f);
        float nh = max(dot(n, h), 0.f);
        float lh = max(dot(l, h), 0.f);
        float D = (nh * nh * (m_2 - 1.f) + 1.f);
        D = m_2 / (D * D * 3.14159265f);
        float Vl = 1.f / (nl + sqrt(m_2 + nl * nl * (1.f - m_2)));
        float Vv = 1.f / (nv + sqrt(m_2 + nv * nv * (1.f - m_2)));
        float g = sqrt(n_2 + lh * lh);
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
            mtr.out[pidx * mtr.channel + k] += intensity * (mtr.diffusemap[pidx * mtr.channel + k] * diffuse + specular);
        }
    }
}

void PBRMaterial::backward(MaterialGradParams& mtr) {
    void* args[] = { &mtr.kernel, &mtr.grad };
    CUDA_ERROR_CHECK(cudaLaunchKernel(PBRMaterialBackwardKernel, mtr.grid, mtr.block, args, 0, NULL));
}