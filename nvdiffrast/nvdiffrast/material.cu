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
    mtr.kernel.diffusemap = in;
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.kernel.out, mtr.Size()));
}

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
    mtr.grad.diffusemap = grad;
    CUDA_ERROR_CHECK(cudaMalloc(&mtr.grad.out, mtr.Size()));
}

void Material::init(MaterialGradParams& mtr, RasterizeParams& rast, ProjectParams& pos, ProjectParams& normal, Attribute* texel, int channel, float* diffusemap, float* roughnessmap, float* normalmap, float* heightmap, float* graddiffuse, float* gradroughness, float* gradnormal, float* gradheight) {
    init((MaterialParams&)mtr, rast, pos, normal, texel, channel, diffusemap, roughnessmap, normalmap, heightmap);
    mtr.grad.diffusemap = graddiffuse;
    mtr.grad.roughnessmap = gradroughness;
    mtr.grad.normalmap = gradnormal;
    mtr.grad.heightmap = gradheight;
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
        mtr.out[pidx * mtr.channel + k] += mtr.diffusemap[pidx * mtr.channel + k] * Ka;
    }
    for (int i = 0; i < mtr.lightNum; i++) {
        float3 l = normalize(mtr.point[i] - pos);
        float ln = dot(l, n);
        float3 r = 2.f * ln * n - l;
        float rv = dot(r, v);
        float powrv = pow(max(rv, 0.f), shininess);
        for (int k = 0; k < mtr.channel; k++) {
            float intensity = mtr.intensity[i * mtr.channel + k];
            float diffuse = intensity * max(ln, 0.f);
            float specular = intensity * powrv;
            mtr.out[pidx * mtr.channel + k] += mtr.diffusemap[pidx * mtr.channel + k] * (Kd * diffuse + Ks * specular);
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
        float din = grad.out[pidx * mtr.channel + k] * mtr.diffusemap[pidx * mtr.channel + k];
        if(grad.params!=nullptr)atomicAdd(&grad.params[0], din);
    }
    for (int i = 0; i < mtr.lightNum; i++) {
        float3 l = ((float3*)mtr.point)[i] - pos;
        float il = 1.f / sqrt(dot(l, l));
        l *= il;
        float ln = dot(l, n);
        float3 r = 2.f * ln * n - l;
        float rv = dot(r, v);

        float powrv = pow(max(rv, 0.f), mtr.params[3]);
        float dl = 0.f;
        for (int k = 0; k < mtr.channel; k++) {
            float dLdout = grad.out[pidx * mtr.channel + k];
            float din = dLdout * mtr.diffusemap[pidx * mtr.channel + k];
            dl += din;
            float intensity = mtr.intensity[i * mtr.channel + k];
            float diffuse = intensity * max(ln, 0.f);
            float specular = intensity * powrv;
            float dshine = din * Ks * specular * log(max(rv, 1e-3));
            dLdout *= (Kd * diffuse + Ks * specular);
            if (grad.diffusemap != nullptr)grad.diffusemap[pidx * mtr.channel + k] += dLdout;
            if (grad.intensity != nullptr)atomicAdd(&grad.intensity[i * mtr.channel + k], dLdout);
            if (grad.params != nullptr) {
                atomicAdd(&grad.params[1], din * diffuse);
                atomicAdd(&grad.params[2], din * specular);
                atomicAdd(&grad.params[3], dshine);
            }
        }
        if (grad.point != nullptr) {
            float3 dLdl = Kd * n;
            if (rv > 0.f)dLdl += Ks * powrv / rv * (2 * dot(n, v) * n - v);
            atomicAdd3(&((float3*)grad.point)[i], dLdl * (make_float3(1.f, 1.f, 1.f) - l * l) * il);
        }
    }
}

void PhongMaterial::backward(MaterialGradParams& mtr) {
    dim3 block = getBlock(mtr.kernel.width, mtr.kernel.height);
    dim3 grid = getGrid(block, mtr.kernel.width, mtr.kernel.height, mtr.kernel.depth);
    void* args[] = { &mtr.kernel, &mtr.grad };
    CUDA_ERROR_CHECK(cudaLaunchKernel(PhongMaterialBackwardKernel, grid, block, args, 0, NULL));
}




//
// Cook-Torrance GGX
// 
// n:normal
// v=normalize(eye-pos)
// l=normalize(light-pos)
// h=normalize(v+l)
// 
// lh=dot(l,h)
// g=sqrt(ior^2-1+lh^2)
// dg/dlh=lh/g
// dg/dior=ior/g
// 
// F0=((g-lh)/(g+lh))^2
// F1=1+((lh*(g+lh)-1)/(lh*(g-lh)+1))^2
// F=1/2*F0*F1
// 
// F'=1/2*(F0'*F1+F0*F1')
// dF/dlh=1/2*4/g*((g-lh)/(g+lh)^3)*(lh^2-g^2)*F1+F0*4/g*(lh^2*ior^2+g^2)*((lh*(g+lh)-1)/(lh*(g-lh)+1)^3)
// dF/dior=1/2*4/g*ior*lh*((g-lh)/(g+lh)^3)*F1+F0*4/g*lh*ior*(1-lh^2)*((lh*(g+lh)-1)/(lh*(g-lh)+1)^3)
// 
// m=roughness^2
// nh=dot(n,h)
// D=m/pi/(nh^2*(m-1)+1)^2
// dD/dnh=-4*m*(m-1)*nh/pi/(nh^2*(m-1)+1)^3
// dD/droughness=2*roughness*(1-nh^2*(m+1))/pi/(nh^2*(m-1)+1)^3
// 
// nv=dot(n,v)
// Vv=1/(nv+sqrt(nv^2+m*(1-nv^2)))
// dVv/dnv=-(sqrt(nv^2+m*(1-nv^2))+nv*(1-m))/sqrt(nv^2*(1-m)+m)/(nv+sqrt(nv^2+m*(1-nv^2)))^2
// dVv/droughness=-2*roughness*(1-nv^2)/sqrt(nv^2+m*(1-nv^2))/(nv+sqrt(nv^2+m*(1-nv^2)))^2
// nl=dot(n,l)
// Vl=1/(nl+sqrt(nl^2+m*(1-nl^2)))
// dVl/dnl=-(sqrt(nl^2+m*(1-nl^2))+nl*(1-m))/sqrt(nl^2*(1-m)+m)/(nl+sqrt(nl^2+m*(1-nl^2)))^2
// dVl/droughness=-2*roughness*(1-nl^2)/sqrt(nl^2+m*(1-nl^2))/(nl+sqrt(nl^2+m*(1-nl^2)))^2
// V=Vv*Vl
// dV/dn=Vv*dVl/dnl*l+Vl*dVv/dnv*v
// dV/dv=Vl*dVv/dnv*n
// dV/dl=Vv*dVl/dnl*n
// dV/droughness=Vv*dVl/droughness+Vl*dVv/droughness
// 
// specular=F*D*Vv*Vl
// diffuse=nl/pi
// ddiffuse/dl=n/pi
// ddiffuse/dn=l/pi
// out=intensity*(in*diffuse+specular)
// dL/dspecular=dL/dout*intensity
// dL/ddiffuse=dL/dout*intensity*in
// dL/din=dL/dout*intensity*diffuse
//

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
    float3 N = normalize(n0 * r.x + n1 * r.y + n2);
    float3 T = normalize(p0 * uv1.y - p1 * uv0.y);
    float3 B = normalize(p1 * uv0.x - p0 * uv1.x);

    float3 normal = ((float3*)mtr.normalmap)[pidx];
    normal = T * normal.x + B * normal.y + N * normal.z;
    float3 n = normalize(normal);
    float3 v = normalize(mtr.eye - pos);

    float m2 = mtr.roughnessmap[pidx] * mtr.roughnessmap[pidx];
    float ior2 = mtr.params[0] * mtr.params[0] - 1.f;
    float disp = mtr.heightmap[pidx];

    float nv = max(dot(n, v), 1e-3);
    float Vv = 1.f / (nv + sqrt(m2 + nv * nv * (1.f - m2)));
    for (int i = 0; i < mtr.lightNum; i++) {
        float3 l = normalize(mtr.point[i] - pos);
        float3 h = normalize(l + v);
        float lh = max(dot(l, h), 0.f);
        float g = sqrt(ior2 + lh * lh);
        float g_nlh = g - lh;
        float g_plh = g + lh;
        float F = (g_nlh / g_plh);
        F *= F * .5f;
        float F1 = (lh * g_plh - 1.f) / (lh * g_nlh + 1.f);
        F *= (1.f + F1 * F1);
        float nh = max(dot(n, h), 0.f);
        float D = (nh * nh * (m2 - 1.f) + 1.f);
        D = m2 / (D * D * 3.14159265f);
        float nl = max(dot(n, l), 1e-3);
        float Vl = 1.f / (nl + sqrt(m2 + nl * nl * (1.f - m2)));
        float specular = D * Vl * Vv * F;
        float diffuse = nl / 3.14159265f;
        for (int k = 0; k < mtr.channel; k++) {
            float intensity = mtr.intensity[i * mtr.channel + k];
            mtr.out[pidx * mtr.channel + k] += intensity * (mtr.diffusemap[pidx * mtr.channel + k] * diffuse + specular);
        }
    }
}

void PBRMaterial::forward(MaterialParams& mtr) {
    CUDA_ERROR_CHECK(cudaMemset(mtr.kernel.out, 0, mtr.Size()));
    dim3 block = getBlock(mtr.kernel.width, mtr.kernel.height);
    dim3 grid = getGrid(block, mtr.kernel.width, mtr.kernel.height, mtr.kernel.depth);
    void* args[] = { &mtr.kernel };
    CUDA_ERROR_CHECK(cudaLaunchKernel(PBRMaterialForwardKernel, grid, block, args, 0, NULL));
}

void PBRMaterial::forward(MaterialGradParams& mtr) {
    CUDA_ERROR_CHECK(cudaMemset(mtr.grad.out, 0, mtr.Size()));
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
    float3 N = normalize(n0 * r.x + n1 * r.y + n2);
    float3 T = normalize(p0 * uv1.y - p1 * uv0.y);
    float3 B = normalize(p1 * uv0.x - p0 * uv1.x);

    float3 normal = ((float3*)mtr.normalmap)[pidx];
    normal = T * normal.x + B * normal.y + N * normal.z;
    float in = 1.f / length(normal);
    float3 n = normal * in;
    float3 v = normalize(mtr.eye - pos);

    float m2 = mtr.roughnessmap[pidx] * mtr.roughnessmap[pidx];
    float m2_1 = 1.f - m2;
    float ior2 = mtr.params[0] * mtr.params[0];

    float nv = max(dot(n, v), 1e-3);
    float v2 = nv * nv;
    float k = sqrt(m2 + v2 * (1.f - m2));
    float Vv = 1.f / (nv + k);
    float kv = Vv * Vv / k;
    float dVvdnv = -(k + nv * m2_1) * kv;
    kv *= (v2 - 1.f);
    float3 dLdn = make_float3(0.f, 0.f, 0.f);
    for (int i = 0; i < mtr.lightNum; i++) {
        float3 l = normalize(mtr.point[i] - pos);
        float3 h = normalize(l + v);

        float nh = max(dot(n, h), 0.f);
        v2 = nh * nh;
        k = 1.f / (1.f - v2 * m2_1);
        float kD = k * k / 3.14159265f;
        float D = m2 * kD;
        kD *= k;
        float dDdnh = 4.f * m2_1 * m2 * nh * kD;
        kD *= (1.f - v2 * (m2 + 1.f));

        float nl = max(dot(n, l), 1e-3);
        v2 = nl * nl;
        k = sqrt(m2 + v2 * (1.f - m2));
        float Vl = 1.f / (nl + k);
        float kl = Vl * Vl / k;
        float dVldnl = -(k + nl * m2_1) * kl;
        kl *= (v2 - 1.f);

        float3 dVdn = dVvdnv * v * Vl + dVldnl * l * Vv;
        float V = Vv * Vl;

        float lh = max(dot(l, h), 0.f);
        v2 = lh * lh;
        float g2 = ior2 - 1.f + v2;
        float g = sqrt(g2);
        float g_plh = g + lh;
        float kF0 = (g - lh) / g_plh;
        float F0 = kF0 * kF0;
        kF0 /= (g_plh * g_plh);
        float g_nlh_1 = lh * (g - lh) + 1.f;
        float kF1 = (lh * g_plh - 1.f) / g_nlh_1;
        float F1 = 1.f + kF1 * kF1;
        float F = .5f * F0 * F1;
        k = 2.f / g;
        float dFdlh = k * (kF0 * (v2 - g2) * F1 + kF1 * (v2 * ior2 + g2) * F0);

        grad.params[0] += k * mtr.params[0] * lh * (kF0 * F1 + kF1 * (1.f - v2) * F0) * D * V;

        float specular = nl * D * Vl * Vv * F;
        float diffuse = nl / 3.14159265f;
        k = kD * V + (kv * Vl + kl * Vv) * D;
        for (int j = 0; j < mtr.channel; j++) {
            float dLdout = grad.out[pidx * mtr.channel + j];
            dLdout *= mtr.intensity[i * mtr.channel + j];
            grad.diffusemap[pidx * mtr.channel + j] += dLdout * diffuse;
            float dLddiffuse = dLdout * mtr.diffusemap[pidx * mtr.channel + j] / 3.14159265f;
            grad.roughnessmap[pidx] += dLdout * k * 2.f * mtr.roughnessmap[pidx];
            dLdn+= dLdout * F * (dDdnh * h * V + dVdn * D) + dLddiffuse * l;
        }
    }
    dLdn = in * (dLdn - n * dot(n, dLdn));
    dLdn = make_float3(dot(dLdn, T), dot(dLdn, B), dot(dLdn, N));
    ((float3*)grad.normalmap)[pidx] += dLdn;
}

void PBRMaterial::backward(MaterialGradParams& mtr) {
    dim3 block = getBlock(mtr.kernel.width, mtr.kernel.height);
    dim3 grid = getGrid(block, mtr.kernel.width, mtr.kernel.height, mtr.kernel.depth);
    if (mtr.kernel.width > mtr.kernel.height) {
        block.x >>= 1; grid.x <<= 1;
    }
    else {
        block.y >>= 1; grid.y <<= 1;
    }
    void* args[] = { &mtr.kernel, &mtr.grad };
    CUDA_ERROR_CHECK(cudaLaunchKernel(PBRMaterialBackwardKernel, grid, block, args, 0, NULL));
}