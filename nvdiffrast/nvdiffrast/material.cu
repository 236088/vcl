#include "material.h"

void NormalAxis::init(NormalAxisParams& norm, RotationParams& rot, RasterizeParams& rast, Attribute& normal) {
    norm.kernel.width = rast.kernel.width;
    norm.kernel.height = rast.kernel.height;
    norm.kernel.depth = rast.kernel.depth;
    norm.kernel.rast = rast.kernel.out;
    norm.kernel.rot = rot.kernel.out;
    norm.kernel.normal = normal.vbo;
    norm.kernel.normalidx = normal.vao;
    CUDA_ERROR_CHECK(cudaMalloc(&norm.kernel.out, norm.Size()));
}

void NormalAxis::init(NormalAxisParams& norm, RotationParams& rot, RasterizeParams& rast, Attribute& normal, Attribute& pos, Attribute& texel, TexturemapParams& normalmap) {
    init(norm, rot, rast, normal);
    if (pos.dimention != 3)ERROR_STRING(dimention is not 3);
    norm.kernel.pos = pos.vbo;
    norm.kernel.posidx = pos.vao;
    norm.kernel.texel = texel.vbo;
    norm.kernel.texelidx = texel.vao;
	if (normalmap.kernel.channel != 3)ERROR_STRING(dimention is not 3);
	norm.kernel.normalmap = normalmap.kernel.out;
}

__global__ void NormalAxisForwardKernel(const NormalAxisKernelParams norm) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= norm.width || py >= norm.height || pz >= norm.depth)return;
    int pidx = px + norm.width * (py + norm.height * pz);

    float4 r = ((float4*)norm.rast)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0) return;

    float3 n2 = ((float3*)norm.normal)[norm.normalidx[idx * 3 + 2]];
    float3 n0 = ((float3*)norm.normal)[norm.normalidx[idx * 3]] - n2;
    float3 n1 = ((float3*)norm.normal)[norm.normalidx[idx * 3 + 1]] - n2;
    float3 N = normalize(n0 * r.x + n1 * r.y + n2);
    if (norm.normalmap == nullptr) {
        float3 normal = make_float3(
            norm.rot[0] * N.x + norm.rot[4] * N.y + norm.rot[8] * N.z,
            -(norm.rot[1] * N.x + norm.rot[5] * N.y + norm.rot[9] * N.z),
            norm.rot[2] * N.x + norm.rot[6] * N.y + norm.rot[10] * N.z
        );
        ((float3*)norm.out)[pidx] = normal;
        return;
    }

    float2 uv2 = ((float2*)norm.texel)[norm.texelidx[idx * 3 + 2]];
    float2 uv0 = ((float2*)norm.texel)[norm.texelidx[idx * 3]] - uv2;
    float2 uv1 = ((float2*)norm.texel)[norm.texelidx[idx * 3 + 1]] - uv2;
    float3 p2 = ((float3*)norm.pos)[norm.posidx[idx * 3 + 2]];
    float3 p0 = ((float3*)norm.pos)[norm.posidx[idx * 3]] - p2;
    float3 p1 = ((float3*)norm.pos)[norm.posidx[idx * 3 + 1]] - p2;

    float3 T = p0 * uv1.y - p1 * uv0.y;
    T = normalize(T - N * dot(N, T));
    float3 B = p1 * uv0.x - p0 * uv1.x;
    B = normalize(B - N * dot(N, B));

    float3 normal = ((float3*)norm.normalmap)[pidx];
    N = normalize(T * normal.x + B * normal.y + N * normal.z);
    ((float3*)norm.out)[pidx] = normalize(make_float3(
        norm.rot[0] * N.x + norm.rot[4] * N.y + norm.rot[8] * N.z,
        -(norm.rot[1] * N.x + norm.rot[5] * N.y + norm.rot[9] * N.z),
        norm.rot[2] * N.x + norm.rot[6] * N.y + norm.rot[10] * N.z
    ));
}

void NormalAxis::forward(NormalAxisParams& norm){
	CUDA_ERROR_CHECK(cudaMemset(norm.kernel.out, 0, norm.Size()));
	dim3 block = getBlock(norm.kernel.width, norm.kernel.height);
	dim3 grid = getGrid(block, norm.kernel.width, norm.kernel.height, norm.kernel.depth);
	void* args[] = { &norm.kernel };
	CUDA_ERROR_CHECK(cudaLaunchKernel(NormalAxisForwardKernel, grid, block, args, 0, NULL));
}



void ViewAxis::init(ViewAxisParams& view, RotationParams& rot, CameraParams& cam, RasterizeParams& rast) {
    view.kernel.width = rast.kernel.width;
    view.kernel.height = rast.kernel.height;
    view.kernel.depth = rast.kernel.depth;
    view.kernel.rot = rot.kernel.out;
    view.kernel.view = cam.kernel.view;
    view.kernel.projection = cam.kernel.projection;
    CUDA_ERROR_CHECK(cudaMalloc(&view.kernel.pvinv, 9 * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&view.kernel.out, view.Size()));
}

__global__ void ViewAxisForwardKernel(const ViewAxisKernelParams view) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= view.width || py >= view.height || pz >= view.depth)return;
    int pidx = px + view.width * (py + view.height * pz);

    float2 screenpos = make_float2(
        (float)px / (float)view.width * 2.f - 1.f,
        (float)py / (float)view.height * 2.f - 1.f
    );

    ((float3*)view.out)[pidx] = normalize(
         ((float3*)view.pvinv)[0] * screenpos.x
        - ((float3*)view.pvinv)[1] * screenpos.y
        + ((float3*)view.pvinv)[2]);
}

void ViewAxis::forward(ViewAxisParams& view) {
    glm::mat3 pvinv = glm::inverse(glm::mat3(*view.kernel.projection) * glm::mat3(*view.kernel.view));
    CUDA_ERROR_CHECK(cudaMemcpy(view.kernel.pvinv, &pvinv, 9 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block = getBlock(view.kernel.width, view.kernel.height);
    dim3 grid = getGrid(block, view.kernel.width, view.kernel.height, view.kernel.depth);
    void* args[] = { &view.kernel};
    CUDA_ERROR_CHECK(cudaLaunchKernel(ViewAxisForwardKernel, grid, block, args, 0, NULL));
}



void SphericalGaussian::init(SphericalGaussianParams& sg, RasterizeParams& rast, NormalAxisParams& normal, ViewAxisParams& view, TexturemapParams& diffuse, TexturemapParams& roughness, SGBuffer& sgbuf, float ior) {
    sg.kernel.width = rast.kernel.width;
    sg.kernel.height = rast.kernel.height;
    sg.kernel.depth = rast.kernel.depth;
    sg.kernel.channel = sgbuf.channel;

    sg.kernel.rast = rast.kernel.out;

    sg.kernel.normal = normal.kernel.out;
    sg.kernel.view = view.kernel.out;
    sg.kernel.diffuse = diffuse.kernel.out;
    sg.kernel.roughness = roughness.kernel.out;
    sg.kernel.ior = ior;

    sg.kernel.sgnum = sgbuf.num;
    sg.kernel.axis = sgbuf.axis;
    sg.kernel.sharpness = sgbuf.sharpness;
    sg.kernel.amplitude = sgbuf.amplitude;

    CUDA_ERROR_CHECK(cudaMalloc(&sg.kernel.out, sg.Size()));
    CUDA_ERROR_CHECK(cudaMalloc(&sg.kernel.outDiffenv, sg.Size()));
    CUDA_ERROR_CHECK(cudaMalloc(&sg.kernel.outSpecenv, sg.Size()));
}


__global__ void SGForwardKernel(const SphericalGaussianKernelParams sg) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= sg.width || py >= sg.height || pz >= sg.depth)return;
    int pidx = px + sg.width * (py + sg.height * pz);

    float3 v = ((float3*)sg.view)[pidx];

    float4 r = ((float4*)sg.rast)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0) {
        for (int i = 0; i < sg.sgnum; i++) {
            float3 sgaxis = ((float3*)sg.axis)[i];
            float sgsharpness = sg.sharpness[i];
            float sgvalue = exp(sgsharpness * (dot(v, sgaxis) - 1.f));
            for (int k = 0; k < sg.channel; k++) {
                AddNaNcheck(sg.out[pidx * sg.channel + k], sg.amplitude[i * sg.channel + k] * sgvalue);
            }
        }
        return;
    }
    v = -v;
    float3 n = ((float3*)sg.normal)[pidx];
    float3 p = 2.f * dot(v, n) * n - v;

    float diffamplitude = .32424871f; //=1/(pi*(1-exp(-4))
    float diffsharpness = 2.f;
    float3 diffn = n * diffsharpness;

    float m = sg.roughness[pidx];
    float m2 = m * m;

    float vn = max(dot(v, n),1e-6);
    float vn2 = vn * vn;
    float Gv = 1.f / (vn + sqrt(vn2 + m2 * (1.f - vn2)));

    float pn = max(dot(p, n),1e-6);
    float pn2 = pn * pn;
    float Gp = 1.f / (pn + sqrt(pn2 + m2 * (1.f - pn2)));

    float g = sqrt(sg.ior * sg.ior - 1.f + pn * pn);
    float g_pls_pn = g + pn;
    float g_mns_pn = g - pn;
    float f0 = g_mns_pn / g_pls_pn;
    float f1 = (pn * g_pls_pn - 1.f) / (pn * g_mns_pn + 1.f);
    float F = .5f * f0 * f0 * (1.f + f1 * f1);

    float specamplitude = F*Gp* Gv;
    float specsharpness = .5f / max(m2 * vn, 1e-6);
    float3 specn = p * specsharpness;

    for (int i = 0; i < sg.sgnum; i++) {
        float sgsharpness = sg.sharpness[i];
        float3 sgn = ((float3*)sg.axis)[i] * sgsharpness;
        float diffl = length(diffn + sgn);
        float diffg = (exp(diffl - diffsharpness - sgsharpness) - exp(-diffl - diffsharpness - sgsharpness)) / max(diffl, 1e-6) * diffamplitude * 6.2831853f;
        float specl = length(specn + sgn);
        float specg = (exp(specl - specsharpness - sgsharpness) - exp(-specl - specsharpness - sgsharpness)) / max(specl, 1e-6) * specamplitude * 6.2831853f;
        for (int k = 0; k < sg.channel; k++) {
            float sgamplitude = sg.amplitude[i * sg.channel + k];
            AddNaNcheck(sg.outDiffenv[pidx * sg.channel + k], diffg * sgamplitude);
            AddNaNcheck(sg.outSpecenv[pidx * sg.channel + k], specg * sgamplitude);
        }
    }

    for (int k = 0; k < sg.channel; k++) {
        int idx = pidx * sg.channel + k;
        sg.out[idx] = sg.diffuse[idx] * sg.outDiffenv[idx] + sg.outSpecenv[idx];
    }
}

void SphericalGaussian::forward(SphericalGaussianParams& sg) {
    CUDA_ERROR_CHECK(cudaMemset(sg.kernel.out, 0, sg.Size()));
    CUDA_ERROR_CHECK(cudaMemset(sg.kernel.outDiffenv, 0, sg.Size()));
    CUDA_ERROR_CHECK(cudaMemset(sg.kernel.outSpecenv, 0, sg.Size()));

    dim3 block = getBlock(sg.kernel.width, sg.kernel.height);
    dim3 grid = getGrid(block, sg.kernel.width, sg.kernel.height, sg.kernel.depth);
    void* args[] = { &sg.kernel};
    CUDA_ERROR_CHECK(cudaLaunchKernel(SGForwardKernel, grid, block, args, 0, NULL));
}

__global__ void SGBackwardKernel(const SphericalGaussianKernelParams sg, const SphericalGaussianGradKernelParams grad) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= sg.width || py >= sg.height || pz >= sg.depth)return;
    int pidx = px + sg.width * (py + sg.height * pz);

    float3 v = ((float3*)sg.view)[pidx];

    float4 r = ((float4*)sg.rast)[pidx];
    int idx = (int)r.w - 1;
    if (idx < 0)  return;

    for (int k = 0; k < sg.channel; k++) {
        int idx = pidx * sg.channel + k;
        grad.diffuse[idx] = grad.out[idx] * sg.outDiffenv[idx];
    }

    float3 n = ((float3*)sg.normal)[pidx];
    float3 p = v - 2.f * dot(v, n) * n;

    float diffamplitude = .32424871f; //=1/(pi*(1-exp(-4))
    float diffsharpness = 2.f;
    float3 diffn = n * diffsharpness;

    float m = sg.roughness[pidx];
    float m2 = m * m;

    float vn = abs(dot(v, n));
    float vn2 = vn * vn;
    float Gv = 1.f / (vn + sqrt(vn2 + m2 * (1.f - vn2)));

    float pn = abs(dot(p, n));
    float pn2 = pn * pn;
    float Gp = 1.f / (pn + sqrt(pn2 + m2 * (1.f - pn2)));

    float g = sqrt(sg.ior * sg.ior - 1.f + pn * pn);
    float g_pls_pn = g + pn;
    float g_mns_pn = g - pn;
    float f0 = g_mns_pn / g_pls_pn;
    float f1 = (pn * g_pls_pn - 1.f) / (pn * g_mns_pn + 1.f);
    float F = .5f * f0 * f0 * (1.f + f1 * f1);

    float specamplitude = F * Gp * Gv;
    float specsharpness = .5f / max(m2 * vn, 1e-6);
    float3 specn = p * specsharpness;

    for (int i = 0; i < sg.sgnum; i++) {
        float sgsharpness = sg.sharpness[i];
        float3 sgn = ((float3*)sg.axis)[i] * sgsharpness;
        float diffl = length(diffn + sgn);
        float diffg = exp(-sgsharpness - diffsharpness) * (exp(diffl) - exp(-diffl)) / max(diffl, 1e-6) * diffamplitude * 6.2831853f;
        float3 specsgn = specn + sgn;
        float specl = length(specsgn);
        float specg = exp(-sgsharpness - specsharpness) * (exp(specl) - exp(-specl)) / max(specl, 1e-6) * specamplitude * 6.2831853f;
        float dgdspecl = exp(-sgsharpness - specsharpness) * ((specl - 1.f) * exp(specl) - (specl + 1.f) * exp(-specl)) / max(dot(specsgn, specsgn), 1e-6) * specamplitude * 6.2831853f;
        float sgamplitudesum = 0.f;
        for (int k = 0; k < sg.channel; k++) {
            sgamplitudesum += sg.amplitude[i * sg.channel + k];
        }

    }
}