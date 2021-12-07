#include "antialias.h"

void Antialias::init(AntialiasParams& aa, RasterizeParams& rast, ProjectParams& proj, float* in, int channel) {
    aa.kernel.width = rast.kernel.width;
    aa.kernel.height = rast.kernel.height;
    aa.kernel.depth = rast.kernel.depth;
    aa.kernel.channel = channel;
    aa.kernel.proj = proj.kernel.out;
    aa.kernel.idx = proj.vao;
    aa.kernel.rast = rast.kernel.out;
    aa.kernel.in = in;
    aa.projNum = proj.kernel.vboNum;
    aa.kernel.xh = rast.kernel.width / 2.f;
    aa.kernel.yh = rast.kernel.height / 2.f;
    CUDA_ERROR_CHECK(cudaMalloc(&aa.kernel.out, aa.Size()));

    aa.block = rast.block;
    aa.grid = rast.grid;
    if (rast.kernel.width > rast.kernel.height) {
        aa.block.x >>= 1;
        aa.grid.x <<= 1;
    }
    else {
        aa.block.y >>= 1;
        aa.grid.y <<= 1;
    }
}

__device__ __forceinline__ void forwardEdgeLeak(const AntialiasKernelParams aa, int pidx, int oidx, float2 pa, float2 pb, float2 o) {
    float a = cross(pa, pb);
    float oa = cross(pa - o, pb - o);
    if (a * oa > 0.f) return;
    float2 e = pa - pb;
    float n = (o.x + o.y) * (o.x - o.y);
    if ((e.x + e.y) * (e.x - e.y) * n > 0.f)return;
    e *= (o.x + o.y);
    float D = n > 0.f ? -e.y : e.x;
    float ia =1.f / (D + (D > 0 ? 1e-3 : -1e-3));
    float alpha = a * ia - .5f;
    for (int i = 0; i < aa.channel; i++) {
        float diff = aa.in[pidx * aa.channel + i] - aa.in[oidx * aa.channel + i];
        if (alpha > 0) {
            atomicAdd(&aa.out[oidx * aa.channel + i], alpha * diff);
        }
        else{
            atomicAdd(&aa.out[pidx * aa.channel + i], alpha * diff);
        }
    }
}

__device__ __forceinline__ void forwardTriangleFetch(const AntialiasKernelParams aa, int pidx, int oidx, float2 f, int d) {
    int idx = (int)aa.rast[pidx * 4 + 3] - 1;
    uint3 tri = ((uint3*)aa.idx)[idx];
    float2 v0 = ((float2*)aa.proj)[tri.x * 2];
    float2 v1 = ((float2*)aa.proj)[tri.y* 2];
    float2 v2 = ((float2*)aa.proj)[tri.z * 2];
    float iw0 =1.f / aa.proj[tri.x * 4 + 3];
    float iw1 =1.f / aa.proj[tri.y * 4 + 3];
    float iw2 =1.f / aa.proj[tri.z * 4 + 3];
    float2 o = make_float2(d - 1, -d);
    if (pidx < oidx) {
        f += o;
        o = -o;
    }
    float2 p0, p1, p2;
    p0.x = (v0.x * iw0 +1.f) * aa.xh - f.x;
    p0.y = (v0.y * iw0 +1.f) * aa.yh - f.y;
    p1.x = (v1.x * iw1 +1.f) * aa.xh - f.x;
    p1.y = (v1.y * iw1 +1.f) * aa.yh - f.y;
    p2.x = (v2.x * iw2 +1.f) * aa.xh - f.x;
    p2.y = (v2.y * iw2 +1.f) * aa.yh - f.y;
    forwardEdgeLeak(aa, pidx, oidx, p0, p1, o);
    forwardEdgeLeak(aa, pidx, oidx, p1, p2, o);
    forwardEdgeLeak(aa, pidx, oidx, p2, p0, o);
}

__global__ void AntialiasForwardKernel(const AntialiasKernelParams aa) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= aa.width || py >= aa.height || pz >= aa.depth)return;
    int pidx = px + aa.width * (py + aa.height * pz);
    for (int i = 0; i < aa.channel; i++) {
        aa.out[pidx * aa.channel + i] = aa.in[pidx * aa.channel + i];
    }
    float2 tri = ((float2*)aa.rast)[pidx * 2 + 1];
    float2 trih = px > 0 ? ((float2*)aa.rast)[(pidx - 1) * 2 + 1] : tri;
    float2 triv = py > 0 ? ((float2*)aa.rast)[(pidx - aa.width) * 2 + 1] : tri;
    float2 f = make_float2((float)px + .5f, (float)py + .5f);

    if (trih.y != tri.y) {
        int oidx = pidx - 1;
        int ppidx = (int)tri.y ? pidx : oidx;
        int opidx = (int)trih.y ? pidx : oidx;
        if (ppidx == opidx) {
            if (tri.x > trih.x)opidx = oidx;
            else ppidx = oidx;
        }
        forwardTriangleFetch(aa, ppidx, opidx, f, 0);
    }
    if (triv.y != tri.y) {
        int oidx = pidx - aa.width;
        int ppidx = (int)tri.y ? pidx : oidx;
        int opidx = (int)triv.y ? pidx : oidx;
        if (ppidx == opidx) {
            if (tri.x > triv.x)opidx = oidx;
            else ppidx = oidx;
        }
        forwardTriangleFetch(aa, ppidx, opidx, f, 1);
    }
}

void Antialias::forward(AntialiasParams& aa) {
    CUDA_ERROR_CHECK(cudaMemset(aa.kernel.out, 0, aa.Size()));
    void* args[] = { &aa.kernel };
    CUDA_ERROR_CHECK(cudaLaunchKernel(AntialiasForwardKernel, aa.grid, aa.block, args, 0, NULL));
}

void Antialias::forward(AntialiasGradParams& aa) {
    CUDA_ERROR_CHECK(cudaMemset(aa.grad.out, 0, aa.Size()));
    forward((AntialiasParams&)aa);
}

void Antialias::init(AntialiasGradParams& aa, RasterizeParams& rast, ProjectGradParams& proj, float* in, float* grad, int channel) {
    init((AntialiasParams&)aa, rast, proj, in, channel);
    CUDA_ERROR_CHECK(cudaMalloc(&aa.grad.out, aa.Size()));
    aa.grad.proj = proj.grad.out;
    aa.grad.in = grad;
}

void Antialias::init(AntialiasGradParams& aa, RasterizeParams& rast, ProjectGradParams& proj, float* in, int channel) {
    init((AntialiasParams&)aa, rast, proj, in, channel);
    CUDA_ERROR_CHECK(cudaMalloc(&aa.grad.out, aa.Size()));
    aa.grad.proj = proj.grad.out;
    aa.grad.in = nullptr;
}

void Antialias::init(AntialiasGradParams& aa, RasterizeParams& rast, ProjectParams& proj, float* in, float* grad, int channel) {
    init((AntialiasParams&)aa, rast, proj, in, channel);
    CUDA_ERROR_CHECK(cudaMalloc(&aa.grad.out, aa.Size()));
    aa.grad.proj = nullptr;
    aa.grad.in = grad;
}

// horizontal :0, vertical 1;
// 
// (pa, pb) crossed edge (va, vb) screen scale and (fx,fy) to orgin
// pa.x = (va.x/va.w+1.f)*width/2 - (0.5+px)
// pa.y = (va.y/va.w+1.f)*height/2 - (0.5+py)
// pb.x = (vb.x/vb.w+1.f)*width/2 - (0.5+px)
// pb.y = (vb.y/vb.w+1.f)*height/2 - (0.5+py)
// 
// if horizontal (1,0)
//   D = pb.y - pa.y
//   d0 = (pa.x*pb.y-pa.y*pb.x) /D
//   d1 = -((pa.x-1)*pb.y-pa.y*(pb.x-1)) /D
//      =1-d0
// if vertical (0,1)
//   D = pa.x - pb.x
//   d0 = (pa.x*pb.y-pa.y*pb.x) /D
//   d1 = -(pa.x*(pb.y-1)-(pa.y-1)*pb.x) /D
//      =1-d0
// D = dot(pa - pb, o);
//
//  
// if d0 - 0.5 > 0 then
//     pout = pin
//     oout = oin + (d0 - 0.5) * (pin - oin)
//     dL/dout = dL/doout
// else then
//     pout = pin + (d0 - 0.5) * (pin - oin)
//     oout = oin
//     dL/dout = dL/dpout
//
// dL/dpin = dL/dpout * dpout/dpin + dL/doout * doout/dpin
//         = dL/dpout + dL/dout * (d0 - 0.5)
// dL/doin = dL/dpout * dpout/doin + dL/doout * doout/doin
//         = dL/doout - dL/dout * (d0 - 0.5)
//
// (f = F/D)' = (F' - f*D')/D
// 
// dL/d(x,y,w) = dL/dpout * dpout/d(x,y,w) + dL/doout * doout/d(x,y,w)
//         =  dL/dout * (pin - oin) * dd0/d(x,y,w) 
// 
// dpa.x/dva.x=width/2/va.w
// dpa.y/dva.y=height/2/va.w
// dpa.x/dva.w=-width/2*va.x/va.w^2=dpa.x/dva.x*pa.x
// dpa.y/dva.w=-height/2*va.y/va.w^2=dpa.y/dva.y*pa.y
// dpb.x/dvb.x=width/2/vb.w
// dpb.y/dvb.y=height/2/vb.w
// dpb.x/dvb.w=-width/2*vb.x/vb.w^2=dpb.x/dvb.x*pb.x
// dpb.y/dvb.w=-height/2*ba.y/vb.w^2=dpb.y/dvb.y*pb.y
// 
// if horizontal
//   dL/dva.x=dL/dout * (pin - oin)/(pb.y - pa.y) * width/2/va.w * pb.y
//   dL/dva.y=dL/dout * (pin - oin)/(pb.y - pa.y) * height/2/va.w * pb.y * (pa.x - pb.x)/(pb.y - pa.y)
//   dL/dvb.x=dL/dout * (pin - oin)/(pb.y - pa.y) * width/2/vb.w * -pa.y
//   dL/dvb.y=dL/dout * (pin - oin)/(pb.y - pa.y) * height/2/vb.w * -pa.y * (pa.x - pb.x)/(pb.y - pa.y)
//   dL/dva.w=-dL/dva.x*pa.x-dL/dva.y*pa.y
//   dL/dvb.w=-dL/dvb.x*pb.x-dL/dvb.y*pb.y
// if vertical
//   dL/dva.x=dL/dout * (pin - oin)/(pa.x - pb.x) * width/2/va.w * -pb.x * (pb.y - pa.y)/(pa.x - pb.x)
//   dL/dva.y=dL/dout * (pin - oin)/(pa.x - pb.x) * height/2/va.w * -pb.x
//   dL/dvb.x=dL/dout * (pin - oin)/(pa.x - pb.x) * width/2/vb.w * pa.x * (pb.y - pa.y)/(pa.x - pb.x)
//   dL/dvb.y=dL/dout * (pin - oin)/(pa.x - pb.x) * height/2/vb.w * pa.x
//   dL/dva.w=-dL/dva.x*pa.x-dL/dva.y*pa.y
//   dL/dvb.w=-dL/dvb.x*pb.x-dL/dvb.y*pb.y
//

__device__ __forceinline__ void backwardEdgeLeak(const AntialiasKernelParams aa, const AntialiasKernelGradParams grad, int pidx, int oidx, float2 pa, float2 pb, int idxa, int idxb, float iwa, float iwb, float2 o) {
    float a = cross(pa, pb);
    float oa = cross(pa - o, pb - o);
    if (a * oa > 0.f) return;
    float2 e = pa - pb;
    float n = (o.x + o.y) * (o.x - o.y);
    if ((e.x + e.y) * (e.x - e.y) * n > 0.f)return;
    e *= (o.x + o.y);
    float D = n > 0.f ? -e.y : e.x;
    float ia = 1.f / (D + (D > 0 ? 1e-3 : -1e-3));
    float alpha = a * ia - .5f;
    float d = 0.f;
    for (int i = 0; i < aa.channel; i++) {
        float dLdout = alpha > 0 ? grad.out[oidx * aa.channel + i] : grad.out[pidx * aa.channel + i];
        float diff = aa.in[pidx * aa.channel + i] - aa.in[oidx * aa.channel + i];
        if (grad.in != nullptr) {
            atomicAdd(&grad.in[pidx * aa.channel + i], alpha * dLdout);
            atomicAdd(&grad.in[oidx * aa.channel + i], -alpha * dLdout);
        }
        d += dLdout * diff;
    }
    if (grad.proj != nullptr) {
        d *= ia;
        float dLdax = d * aa.xh * iwa;
        float dLday = d * aa.yh * iwa;
        float dLdbx = d * aa.xh * iwb;
        float dLdby = d * aa.yh * iwb;

        float r = n > 0 ? e.x : -e.y;
        r *= ia;
        if (n > 0) {
            dLdax *= pb.y;
            dLday *= pb.y * r;
            dLdbx *= -pa.y;
            dLdby *= -pa.y * r;
        }
        else {
            dLdax *= -pb.x * r;
            dLday *= -pb.x;
            dLdbx *= pa.x * r;
            dLdby *= pa.x;

        }
        atomicAdd_xyw(grad.proj + idxa * 4, dLdax, dLday, -dLdax * pa.x - dLday * pa.y);
        atomicAdd_xyw(grad.proj + idxb * 4, dLdbx, dLdby, -dLdbx * pb.x - dLdby * pb.y);
    }
}

__device__ __forceinline__ void backwardTriangleFetch(const AntialiasKernelParams aa, const AntialiasKernelGradParams grad, int pidx, int oidx, float2 f, int d) {
    int idx = (int)aa.rast[pidx * 4 + 3] - 1;
    uint3 tri = ((uint3*)aa.idx)[idx];
    float2 v0 = ((float2*)aa.proj)[tri.x * 2];
    float2 v1 = ((float2*)aa.proj)[tri.y * 2];
    float2 v2 = ((float2*)aa.proj)[tri.z * 2];
    float iw0 = 1.f / aa.proj[tri.x * 4 + 3];
    float iw1 = 1.f / aa.proj[tri.y * 4 + 3];
    float iw2 = 1.f / aa.proj[tri.z * 4 + 3];
    float2 o = make_float2(d - 1, -d);
    if (pidx < oidx) {
        f += o;
        o = -o;
    }
    float2 p0, p1, p2;
    p0.x = (v0.x * iw0 +1.f) * aa.xh - f.x;
    p0.y = (v0.y * iw0 +1.f) * aa.yh - f.y;
    p1.x = (v1.x * iw1 +1.f) * aa.xh - f.x;
    p1.y = (v1.y * iw1 +1.f) * aa.yh - f.y;
    p2.x = (v2.x * iw2 +1.f) * aa.xh - f.x;
    p2.y = (v2.y * iw2 +1.f) * aa.yh - f.y;

    backwardEdgeLeak(aa, grad, pidx, oidx, p0, p1, tri.x, tri.y, iw0, iw1, o);
    backwardEdgeLeak(aa, grad, pidx, oidx, p1, p2, tri.y, tri.z, iw1, iw2, o);
    backwardEdgeLeak(aa, grad, pidx, oidx, p2, p0, tri.z, tri.x, iw2, iw0, o);
}

__global__ void AntialiasBackwardKernel(const AntialiasKernelParams aa, const AntialiasKernelGradParams grad) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= aa.width || py >= aa.height || pz >= aa.depth)return;
    int pidx = px + aa.width * (py + aa.height * pz);
    for (int i = 0; i < aa.channel; i++) {
        grad.in[pidx * aa.channel + i] = grad.out[pidx * aa.channel + i];
    }

    float2 tri = ((float2*)aa.rast)[pidx * 2 + 1];
    float2 trih = px > 0 ? ((float2*)aa.rast)[(pidx - 1) * 2 + 1] : tri;
    float2 triv = py > 0 ? ((float2*)aa.rast)[(pidx - aa.width) * 2 + 1] : tri;
    float2 f = make_float2((float)px + .5f, (float)py + .5f);

    if (trih.y != tri.y) {
        int oidx = pidx - 1;
        int ppidx = (int)tri.y ? pidx : oidx;
        int opidx = (int)tri.y ? pidx : oidx;
        if (ppidx == opidx) {
            if (tri.x > trih.x)opidx = oidx;
            else ppidx = oidx;
        }
        backwardTriangleFetch(aa, grad, ppidx, opidx, f, 0);
    }
    if (triv.y != tri.y) {
        int oidx = pidx - aa.width;
        int ppidx = (int)tri.y ? pidx : oidx;
        int opidx = (int)triv.y ? pidx : oidx;
        if (ppidx == opidx) {
            if (tri.x > triv.x)opidx = oidx;
            else ppidx = oidx;
        }
        backwardTriangleFetch(aa, grad, ppidx, opidx, f, 1);
    }
}

void Antialias::backward(AntialiasGradParams& aa) {
    void* args[] = { &aa.kernel,&aa.grad };
    CUDA_ERROR_CHECK(cudaLaunchKernel( AntialiasBackwardKernel, aa.grid, aa.block , args, 0, NULL));
}