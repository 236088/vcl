#include "buffer.h"

__global__ void setValueKernel(float* dst, int width, int height, int dimention, float* color) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height)return;
    int pidx = px + width * py;

    for (int i = 0; i < dimention; i++) {
        dst[pidx * dimention + i] = color[i];
    }
}

__global__ void randomKernel(float* dst, float min, float max, int width, int height, int dimention, unsigned int seed) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= width || py >= height || pz >= dimention)return;
    int pidx = px + width * (py + height * pz);
    dst[pidx] += min + (max - min) * getUniform(pidx, seed, 0xba5ec0de);
}

__global__ void sphericalRandomKernel(float* dst, int width, int height, unsigned int seed) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height)return;
    int pidx = px + width * py;

    float theta = (getUniform(pidx, seed, 0xba5eba11) * 2.f - 1.f) * 3.14159265f;
    float z = getUniform(pidx, seed, 0xdeadba11) * 2.f - 1.f;
    float c = sqrt(1.f - z * z);

    dst[pidx * 3] = c * cos(theta);
    dst[pidx * 3 + 1] = c * sin(theta);
    dst[pidx * 3 + 2] = z;
}

__global__ void linerKernel(float* dst, float* src, float w, float b, int width, int height, int dimention) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= width || py >= height || pz >= dimention)return;
    int pidx = px + width * (py + height * pz);
    dst[pidx] = src[pidx] * w + b;
}

__global__ void powexpKernel(float* dst, float* src, float c, float base, int width, int height, int dimention) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= width || py >= height || pz >= dimention)return;
    int pidx = px + width * (py + height * pz);
    dst[pidx] = c * pow(base, src[pidx]);
}

__global__ void powbaseKernel(float* dst, float* src, float c, float exp, int width, int height, int dimention) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= width || py >= height || pz >= dimention)return;
    int pidx = px + width * (py + height * pz);
    dst[pidx] = c * pow(src[pidx], exp);
}

__global__ void clampKernel(float* dst, float* src, float min, float max, int width, int height, int dimention) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= width || py >= height || pz >= dimention)return;
    int pidx = px + width * (py + height * pz);
    dst[pidx] = clamp(src[pidx], min, max);
}

__global__ void stepKernel(float* dst, float* src, float threshold, int width, int height, int dimention) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= width || py >= height || pz >= dimention)return;
    int pidx = px + width * (py + height * pz);
    dst[pidx] = src[pidx] < threshold ? 0.f : 1.f;
}

__global__ void normalizeKernel(float* dst, float* src, int width, int height, int dimention) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height)return;
    int pidx = px + width * py;
    float l2 = 0.f;
    for (int i = 0; i < dimention; i++) {
        l2 += src[pidx * dimention + i] * src[pidx * dimention + i];
    }
    if (l2 <= 0)return;
    float il = 1.f / sqrt(l2);
    for (int i = 0; i < dimention; i++) {
        dst[pidx * dimention + i] *= il;
    }
}

__global__ void sigmoidKernel(float* dst, float* src, float alpha, int width, int height, int dimention) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= width || py >= height || pz >= dimention)return;
    int pidx = px + width * (py + height * pz);

    for (int i = 0; i < dimention; i++) {
        dst[pidx * dimention + i] = 1.f / (1.f + exp(-alpha * src[pidx * dimention + i]));
    }
}



void Buffer::init(Buffer& buf, int num, int dimention) {
    buf.num = num;
    buf.dimention = dimention;
    CUDA_ERROR_CHECK(cudaMalloc(&buf.buffer, buf.Size()));
}

void Buffer::init(Buffer& buf, Buffer& src, int dimention) {
    Buffer::init(buf, src.num, dimention);
}

void Buffer::copy(Buffer& dst, Buffer& src) {
    CUDA_ERROR_CHECK(cudaMemcpy(dst.buffer, src.buffer, dst.Size(), cudaMemcpyDeviceToDevice));
}

void Buffer::copy(Buffer& dst, float* src) {
    CUDA_ERROR_CHECK(cudaMemcpy(dst.buffer, src, dst.Size(), cudaMemcpyHostToDevice));
}

void Buffer::copy(float* dst, Buffer& src) {
    CUDA_ERROR_CHECK(cudaMemcpy(dst, src.buffer, src.Size(), cudaMemcpyDeviceToHost));
}




void Buffer::liner(Buffer& buf, float w, float b) {
    int dimention = 1;
    dim3 block = getBlock(buf.num, buf.dimention);
    dim3 grid = getGrid(block, buf.num, buf.dimention);
    void* args[] = { &buf.buffer, &buf.buffer, &w, &b, &buf.num, &buf.dimention, &dimention};
    CUDA_ERROR_CHECK(cudaLaunchKernel(linerKernel, grid, block, args, 0, NULL));
}

void Buffer::addRandom(Buffer& buf, float min, float max) {
    int dimention = 1;
    unsigned int seed = rand();
    dim3 block = getBlock(buf.num, buf.dimention);
    dim3 grid = getGrid(block, buf.num, buf.dimention);
    void* args[] = { &buf.buffer, &min, &max, &buf.num, &buf.dimention, &dimention, &seed };
    CUDA_ERROR_CHECK(cudaLaunchKernel(randomKernel, grid, block, args, 0, NULL));
}

void Buffer::clamp(Buffer& buf, float min, float max) {
    int dimention = 1;
    unsigned int seed = rand();
    dim3 block = getBlock(buf.num, buf.dimention);
    dim3 grid = getGrid(block, buf.num, buf.dimention);
    void* args[] = {&buf.buffer, &buf.buffer, &min, &max, &buf.num, &buf.dimention, &dimention};
    CUDA_ERROR_CHECK(cudaLaunchKernel(clampKernel, grid, block, args, 0, NULL));
}

void Buffer::normalize(Buffer& buf) {
    int height = 1;
    dim3 block = getBlock(buf.num, height);
    dim3 grid = getGrid(block, buf.num, height);
    void* args[] = {&buf.buffer, &buf.buffer, &buf.num, &height, &buf.dimention };
    CUDA_ERROR_CHECK(cudaLaunchKernel(normalizeKernel, grid, block, args, 0, NULL));
}

void BufferGrad::init(BufferGrad& buf, int num, int dimention) {
    Buffer::init(buf, num, dimention);
    CUDA_ERROR_CHECK(cudaMalloc(&buf.grad, buf.Size()));
}

void BufferGrad::init(BufferGrad& buf, Buffer& src, int dimention) {
    init(buf, src.num, dimention);
}

void BufferGrad::clear(BufferGrad& buf) {
    CUDA_ERROR_CHECK(cudaMemset(buf.grad, 0, buf.Size()));
}



void Attribute::init(Attribute& attr, int vboNum, int vaoNum, int dimention) {
    attr.dimention = dimention;
    attr.vboNum = vboNum;
    attr.vaoNum = vaoNum;
    CUDA_ERROR_CHECK(cudaMalloc(&attr.vbo, attr.vboSize()));
    CUDA_ERROR_CHECK(cudaMalloc(&attr.vao, attr.vaoSize()));
}

void Attribute::init(Attribute& attr, Attribute& src, int dimention) {
    attr.dimention = dimention;
    attr.vboNum = src.vboNum;
    CUDA_ERROR_CHECK(cudaMalloc(&attr.vbo, attr.vboSize()));
    attr.vaoNum = src.vaoNum;
    attr.vao = src.vao;
}

void Attribute::loadOBJ(const char* path, Attribute* pos, Attribute* texel, Attribute* normal) {
    FILE* file = fopen(path, "r");
    if (file == NULL) {
        printf("Impossible to open the file !\n");
        return;
    }

    vector<float> tempPos, tempTexel, tempNormal;
    vector<unsigned int> tempPosIndex, tempTexelIndex, tempNormalIndex;
    int posNum = 0, texelNum = 0, normalNum = 0, indexNum = 0;
    while (1) {
        char lineHeader[128];
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF)
            break;
        if (strcmp(lineHeader, "v") == 0) {
            float v[3];
            fscanf(file, "%f %f %f\n", &v[0], &v[1], &v[2]);
            tempPos.push_back(v[0]);
            tempPos.push_back(v[1]);
            tempPos.push_back(v[2]);
            posNum++;
        }
        else if (strcmp(lineHeader, "vt") == 0) {
            float v[2];
            fscanf(file, "%f %f\n", &v[0], &v[1]);
            tempTexel.push_back(v[0]);
            tempTexel.push_back(v[1]);
            texelNum++;
        }
        else if (strcmp(lineHeader, "vn") == 0) {
            float v[3];
            fscanf(file, "%f %f %f\n", &v[0], &v[1], &v[2]);
            tempNormal.push_back(v[0]);
            tempNormal.push_back(v[1]);
            tempNormal.push_back(v[2]);
            normalNum++;
        }
        else if (strcmp(lineHeader, "f") == 0 && posNum > 0) {
            unsigned int idx[9];
            if (texelNum > 0 && normalNum > 0) {
                int matches = fscanf(file, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &idx[0], &idx[3], &idx[6], &idx[1], &idx[4], &idx[7], &idx[2], &idx[5], &idx[8]);
                if (matches != 9) {
                    printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                    return;
                }
                tempTexelIndex.push_back(idx[3] - 1);
                tempTexelIndex.push_back(idx[4] - 1);
                tempTexelIndex.push_back(idx[5] - 1);
                tempNormalIndex.push_back(idx[6] - 1);
                tempNormalIndex.push_back(idx[7] - 1);
                tempNormalIndex.push_back(idx[8] - 1);
            }
            else if (texelNum > 0) {
                int matches = fscanf(file, "%d/%d %d/%d %d/%d\n", &idx[0], &idx[3], &idx[1], &idx[4], &idx[2], &idx[5]);
                if (matches != 6) {
                    printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                    return;
                }
                tempTexelIndex.push_back(idx[3] - 1);
                tempTexelIndex.push_back(idx[4] - 1);
                tempTexelIndex.push_back(idx[5] - 1);
            }
            else if (normalNum > 0) {
                int matches = fscanf(file, "%d//%d %d//%d %d//%d\n", &idx[0], &idx[6], &idx[1], &idx[7], &idx[2], &idx[8]);
                if (matches != 6) {
                    printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                    return;
                }
                tempNormalIndex.push_back(idx[6] - 1);
                tempNormalIndex.push_back(idx[7] - 1);
                tempNormalIndex.push_back(idx[8] - 1);
            }
            else {
                int matches = fscanf(file, "%d %d %d\n", &idx[0], &idx[1], &idx[2]);
                if (matches != 3) {
                    printf("File can't be read by our simple parser : ( Try exporting with other options\n");
                    return;
                }
            }
            tempPosIndex.push_back(idx[0] - 1);
            tempPosIndex.push_back(idx[1] - 1);
            tempPosIndex.push_back(idx[2] - 1);
            indexNum++;
        }
    }


    if (posNum > 0 && pos != nullptr) {
        Attribute::init(*pos, posNum, indexNum, 3);
        CUDA_ERROR_CHECK(cudaMemcpy(pos->vbo, tempPos.data(), pos->vboSize(), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(pos->vao, tempPosIndex.data(), pos->vaoSize(), cudaMemcpyHostToDevice));
    }
    if (texelNum > 0 && texel != nullptr) {
        Attribute::init(*texel, texelNum, indexNum, 2);
        CUDA_ERROR_CHECK(cudaMemcpy(texel->vbo, tempTexel.data(), texel->vboSize(), cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(texel->vao, tempTexelIndex.data(), texel->vaoSize(), cudaMemcpyHostToDevice));
    }
    if (normalNum > 0 && normal != nullptr) {
        Attribute::init(*normal, normalNum, indexNum, 3);
        CUDA_ERROR_CHECK(cudaMemcpy(normal->vbo, tempNormal.data(), normal->vboSize(), cudaMemcpyHostToDevice));;
        CUDA_ERROR_CHECK(cudaMemcpy(normal->vao, tempNormalIndex.data(), normal->vaoSize(), cudaMemcpyHostToDevice));
    }
}

void Attribute::copy(Attribute& dst, Attribute& src) {
    cudaMemcpy(dst.vbo, src.vbo, dst.vboSize(), cudaMemcpyDeviceToDevice);
    if(dst.vao!=src.vao)cudaMemcpy(dst.vao, src.vao, dst.vaoSize(), cudaMemcpyDeviceToDevice);
}

void Attribute::liner(Attribute& attr, float w, float b) {
    int dimention = 1;
    dim3 block = getBlock(attr.vboNum, attr.dimention);
    dim3 grid = getGrid(block, attr.vboNum, attr.dimention);
    void* args[] = { &attr.vbo, &attr.vbo, &w, &b, &attr.vboNum, &attr.dimention, &dimention};
    CUDA_ERROR_CHECK(cudaLaunchKernel(linerKernel, grid, block, args, 0, NULL));
}

__global__ void distanceErrorKernel(const Attribute predict, const Attribute target, float* sum) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= predict.vboNum)return;
    float d = 0.f;
    for (int i = 0; i < predict.dimention; i++) {
        float diff = predict.vbo[px * predict.dimention + i] - target.vbo[px * target.dimention + i];
        d += diff * diff;
    }
    atomicAdd(sum, sqrt(d));
}

float Attribute::distanceError(Attribute& predict, Attribute& target) {
    dim3 block = getBlock(predict.vboNum, 1);
    dim3 grid = getGrid(block, predict.vboNum, 1);
    float* dev;
    CUDA_ERROR_CHECK(cudaMalloc(&dev, sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemset(dev, 0, sizeof(float)));
    void* args[] = { &predict, &target, &dev};
    CUDA_ERROR_CHECK(cudaLaunchKernel(distanceErrorKernel, grid, block, args, 0, NULL));
    float sum;
    CUDA_ERROR_CHECK(cudaMemcpy(&sum, dev, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_ERROR_CHECK(cudaFree(dev));
    return sum / float(predict.vboNum);
}

void Attribute::addRandom(Attribute& attr, float min, float max) {
    int dimention = 1;
    unsigned int seed = rand();
    dim3 block = getBlock(attr.vboNum, attr.dimention);
    dim3 grid = getGrid(block, attr.vboNum, attr.dimention);
    void* args[] = { &attr.vbo, &min, &max, &attr.vboNum, &attr.dimention, &dimention, &seed };
    CUDA_ERROR_CHECK(cudaLaunchKernel(randomKernel, grid, block, args, 0, NULL));
}

void Attribute::step(Attribute& attr, float threshold) {
    int dimention = 1;
    dim3 block = getBlock(attr.vboNum, attr.dimention);
    dim3 grid = getGrid(block, attr.vboNum, attr.dimention);
    void* args[] = { &attr.vbo, &attr.vbo, &threshold,&attr.vboNum, &attr.dimention, &dimention};
    CUDA_ERROR_CHECK(cudaLaunchKernel(stepKernel, grid, block, args, 0, NULL));
}

void AttributeGrad::init(AttributeGrad& attr, int vboNum, int vaoNum, int dimention) {
    Attribute::init(attr, vboNum, vaoNum, dimention);
    CUDA_ERROR_CHECK(cudaMalloc(&attr.grad, attr.vboSize()));
}

void AttributeGrad::init(AttributeGrad& attr, Attribute& src, int dimention) {
    Attribute::init(attr, src, dimention);
    CUDA_ERROR_CHECK(cudaMalloc(&attr.grad, attr.vboSize()));
}

void AttributeGrad::clear(AttributeGrad& attr) {
    CUDA_ERROR_CHECK(cudaMemset(attr.grad, 0, attr.vboSize()));
}



void Texture::init(Texture& texture, int width, int height, int channel, int miplevel){
    int maxlevel = LSB(width | height) + 1;
    if (maxlevel > TEX_MAX_MIP_LEVEL)maxlevel = TEX_MAX_MIP_LEVEL;
    texture.width = width;
    texture.height = height;
    texture.channel = channel;
    texture.miplevel = miplevel < 1 ? 1 : (maxlevel < miplevel ? maxlevel : miplevel);
    int w = width, h = height;
    for (int i = 0; i < texture.miplevel; i++) {
        CUDA_ERROR_CHECK(cudaMalloc(&texture.texture[i], (size_t)w * h * channel * sizeof(float)));
        w >>= 1; h >>= 1;
    }
};

void Texture::init(Texture& texture, float* original, int width, int height, int channel, int miplevel){
    int maxlevel = LSB(width | height) + 1;
    if (maxlevel > TEX_MAX_MIP_LEVEL)maxlevel = TEX_MAX_MIP_LEVEL;
    texture.width = width;
    texture.height = height;
    texture.channel = channel;
    texture.miplevel = miplevel < 1 ? 1 : (maxlevel < miplevel ? maxlevel : miplevel);
    texture.texture[0] = original;
    int w = width, h = height;
    for (int i = 1; i < texture.miplevel; i++) {
        CUDA_ERROR_CHECK(cudaMalloc(&texture.texture[i], (size_t)w * h * channel * sizeof(float)));
        w >>= 1; h >>= 1;
    }
    buildMIP(texture);
};


__global__ void bilinearDownsamplingkernel(const Texture texture, int index, int width, int height) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= width || py >= height)return;
    int pidx = px + width * (py + height * pz);
    px <<= 1; py <<= 1;
    width <<= 1; height <<= 1;
    int xs = px > 0 ? -1 : 0;
    int xe = px < width - 2 ? 2 : 1;
    int ys = py > 0 ? -1 : 0;
    int ye = py < height - 2 ? 2 : 1;

    int idx = index - 1;
    float filter[4] = { .125f,.375f,.375f,.125f };
    for (int i = 0; i < 3; i++)texture.texture[index][pidx * texture.channel + i] = 0.f;
    for (int x = xs; x <= xe; x++) {
        for (int y = ys; y <= ye; y++) {
            float f = filter[x + 1] * filter[y + 1];
            int p = (px + x) + width * (py + y);
            for (int i = 0; i < 3; i++) {
                texture.texture[index][pidx * texture.channel + i] += texture.texture[idx][p * texture.channel + i] * f;
            }
        }
    }
}

void Texture::bilinearDownsampling(Texture& texture) {
    int i = 1;
    int w = texture.width, h = texture.height;
    void* args[] = { &texture, &i, &w, &h };
    for (; i < texture.miplevel; i++) {
        w >>= 1; h >>= 1;
        dim3 block = getBlock(w, h);
        dim3 grid = getGrid(block, w, h);
        CUDA_ERROR_CHECK(cudaLaunchKernel(bilinearDownsamplingkernel, grid, block, args, 0, NULL));
    }
}

__global__ void downSampling(const Texture texture, int index, int width, int height) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= width || py >= height)return;
    int pidx = px + width * (py + height * pz);
    px <<= 1; py <<= 1;
    width <<= 1; height <<= 1;
    int p00idx = px + width * (py + height * pz);
    int p01idx = p00idx + 1;
    int p10idx = p00idx + width;
    int p11idx = p10idx + 1;

    int idx = index - 1;
    for (int i = 0; i < texture.channel; i++) {
        float p00 = texture.texture[idx][p00idx * texture.channel + i];
        float p01 = texture.texture[idx][p01idx * texture.channel + i];
        float p10 = texture.texture[idx][p10idx * texture.channel + i];
        float p11 = texture.texture[idx][p11idx * texture.channel + i];

        float p = (p00 + p01 + p10 + p11) * 0.25f;
        texture.texture[index][pidx * texture.channel + i] = p;
    }
}

void Texture::buildMIP(Texture& texture) {
    int i = 1;
    int w = texture.width, h = texture.height;
    void* args[] = { &texture, &i, &w, &h };
    for (; i < texture.miplevel; i++) {
        w >>= 1; h >>= 1;
        dim3 block = getBlock(w, h);
        dim3 grid = getGrid(block, w, h);
        CUDA_ERROR_CHECK(cudaLaunchKernel(downSampling, grid, block, args, 0, NULL));
    }
}

__global__ void bmpUcharToFloat(unsigned char* data, const Texture texture) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= texture.width || py >= texture.height)return;
    int pidx = px + texture.width * (py + texture.height * pz);

    for (int i = 0; i < texture.channel; i++) {
        texture.texture[0][pidx * texture.channel + i] = (float)data[(pidx + 1) * texture.channel - (i + 1)] / 255.f;
    }
}

void Texture::loadBMP(const char* path, Texture& texture, int miplevel) {
    unsigned char header[54];

    FILE* file = fopen(path, "rb");
    if (!file) {
        ERROR_STRING(Image could not be opened);
        return;
    }
    if (fread(header, 1, 54, file) != 54) {
        ERROR_STRING(Not a correct BMP file);
        return;
    }
    if (header[0] != 'B' || header[1] != 'M') {
        ERROR_STRING(Not a correct BMP file);
        return;
    }
    unsigned int dataPos = *(int*)&(header[0x0A]);
    unsigned int imageSize = *(int*)&(header[0x22]);
    unsigned int width = *(int*)&(header[0x12]);
    unsigned int height = *(int*)&(header[0x16]);
    unsigned int channel = *(int*)&(header[0x1c]) / 8;
    Texture::init(texture, width, height, channel, miplevel);
    if (imageSize == 0)    imageSize = width * height * channel;
    if (dataPos == 0)      dataPos = 54;
    fseek(file, dataPos, SEEK_SET);

    unsigned char* data;
    cudaMallocHost(&data, imageSize * sizeof(unsigned char));
    fread(data, 1, imageSize, file);
    fclose(file);

    unsigned char* dev_data;

    CUDA_ERROR_CHECK(cudaMalloc(&dev_data, (size_t)imageSize * sizeof(unsigned char)));
    CUDA_ERROR_CHECK(cudaMemcpy(dev_data, data, (size_t)imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice));

    dim3 block = getBlock(width, height);
    dim3 grid = getGrid(block, width, height);
    void* args[] = { &dev_data,&texture };
    CUDA_ERROR_CHECK(cudaLaunchKernel(bmpUcharToFloat, grid, block, args, 0, NULL));
    CUDA_ERROR_CHECK(cudaFreeHost(data));
    CUDA_ERROR_CHECK(cudaFree(dev_data));
    buildMIP(texture);
}

void Texture::setColor(Texture& texture, float* color) {
    int width = texture.width, height = texture.height;
    float* dev_color;
    CUDA_ERROR_CHECK(cudaMalloc(&dev_color, (size_t)texture.channel * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(dev_color, color, (size_t)texture.channel * sizeof(float), cudaMemcpyHostToDevice));
    for (int i = 0; i < texture.miplevel; i++) {
        void* args[] = { &texture.texture[i], &width, &height ,&texture.channel, &dev_color };
        dim3 block = getBlock(width, height);
        dim3 grid = getGrid(block, width, height);
        CUDA_ERROR_CHECK(cudaLaunchKernel(setValueKernel, grid, block, args, 0, NULL));
        width >>= 1; height >>= 1;
    }
    CUDA_ERROR_CHECK(cudaFree(dev_color));
}

void Texture::liner(Texture& texture, float w, float b) {
    int width = texture.width, height = texture.height;
    for (int i = 0; i < texture.miplevel; i++, width >>= 1, height >>= 1) {
        void* args[] = { &texture.texture[i], &texture.texture[i], &w, &b, &width, &height, &texture.channel};
        dim3 block = getBlock(width, height);
        dim3 grid = getGrid(block, width, height, texture.channel);
        CUDA_ERROR_CHECK(cudaLaunchKernel(linerKernel, grid, block, args, 0, NULL));
    }
}

void Texture::normalize(Texture& texture) {
    int width = texture.width, height = texture.height;
    for (int i = 0; i < texture.miplevel; i++, width >>= 1, height >>= 1) {
        dim3 block = getBlock(width, height);
        dim3 grid = getGrid(block, width, height);
        void* args[] = { &texture.texture[i], &texture.texture[i], &width, &height, &texture.channel};
        CUDA_ERROR_CHECK(cudaLaunchKernel(normalizeKernel, grid, block, args, 0, NULL));
    }
}

void Texture::addRandom(Texture& texture, float max, float min) {
    int dimention = 1;
    unsigned int seed = rand();
    dim3 block = getBlock(texture.width, texture.height);
    dim3 grid = getGrid(block, texture.width, texture.height, texture.channel);
    void* args[] = { &texture.texture[0], &max, &min ,&texture.width, &texture.height, &dimention, &seed };
    CUDA_ERROR_CHECK(cudaLaunchKernel(randomKernel, grid, block, args, 0, NULL));
    buildMIP(texture);
}

void Texture::clamp(Texture& texture, float min, float max) {
    dim3 block = getBlock(texture.width, texture.height);
    dim3 grid = getGrid(block, texture.width, texture.height, texture.channel);
    void* args[] = { &texture.texture[0], &texture.texture[0], &min ,&max, &texture.width, &texture.height, &texture.channel};
    CUDA_ERROR_CHECK(cudaLaunchKernel(clampKernel, grid, block, args, 0, NULL));
    buildMIP(texture);
}



void TextureGrad::init(TextureGrad& texture, int width, int height, int channel, int miplevel) {
    Texture::init(texture, width, height, channel, miplevel);
    int w = width, h = height;
    for (int i = 0; i < miplevel; i++) {
        CUDA_ERROR_CHECK(cudaMalloc(&texture.grad[i], (size_t)w * h * channel * sizeof(float)));
        w >>= 1; h >>= 1;
    }
};

void TextureGrad::clear(TextureGrad& texture) {
    int w = texture.width, h = texture.height;
    for (int i = 0; i < texture.miplevel; i++) {
        CUDA_ERROR_CHECK(cudaMemset(texture.grad[i], 0, (size_t)w * h * texture.channel * sizeof(float)));
        w >>= 1; h >>= 1;
    }
}

__global__ void gardAddThrough(const TextureGrad texture, int index, int width, int height) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= width || py >= height)return;
    int pidx = px + width * (py + height * pz);
    px <<= 1; py <<= 1;
    width <<= 1; height <<= 1;
    int p00idx = px + width * (py + height * pz);
    int p01idx = p00idx + 1;
    int p10idx = p00idx + width;
    int p11idx = p10idx + 1;

    int idx = index - 1;
    for (int i = 0; i < texture.channel; i++) {
        float g = texture.grad[index][pidx * texture.channel + i];
        AddNaNcheck(texture.grad[idx][p00idx * texture.channel + i], g);
        AddNaNcheck(texture.grad[idx][p01idx * texture.channel + i], g);
        AddNaNcheck(texture.grad[idx][p10idx * texture.channel + i], g);
        AddNaNcheck(texture.grad[idx][p11idx * texture.channel + i], g);
    }
}

void TextureGrad::gradSumup(TextureGrad& texture) {
    int i = 0;
    int w = texture.width >> (texture.miplevel - 1); int h = texture.height >> (texture.miplevel - 1);
    void* args[] = { &texture, &i, &w, &h };
    for (i = texture.miplevel - 1; i > 0; i--) {
        dim3 block = getBlock(w, h);
        dim3 grid = getGrid(block, w, h);
        CUDA_ERROR_CHECK(cudaLaunchKernel(gardAddThrough, grid, block, args, 0, NULL));
        w <<= 1; h <<= 1;
    }
}



void SGBuffer::init(SGBuffer& sgbuf, int num, int channel) {
    sgbuf.num = num;
    sgbuf.channel = channel;
    CUDA_ERROR_CHECK(cudaMalloc(&sgbuf.axis, (size_t)num * 3 * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&sgbuf.sharpness, (size_t)num * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&sgbuf.amplitude, (size_t)num * channel * sizeof(float)));
}

void SGBuffer::copy(SGBuffer& dst, float* axis, float* sharpness, float* amplitude) {
    if (axis != nullptr)
        CUDA_ERROR_CHECK(cudaMemcpy(dst.axis, axis, (size_t)dst.num * 3 * sizeof(float), cudaMemcpyHostToDevice));
    if (sharpness != nullptr)
        CUDA_ERROR_CHECK(cudaMemcpy(dst.sharpness, sharpness, (size_t)dst.num * sizeof(float), cudaMemcpyHostToDevice));
    if (amplitude != nullptr)
        CUDA_ERROR_CHECK(cudaMemcpy(dst.amplitude, amplitude, (size_t)dst.num * dst.channel * sizeof(float), cudaMemcpyHostToDevice));
}

void SGBuffer::randomize(SGBuffer& sgbuf) {
    int dimention = 1;
    unsigned int seed = rand();
    dim3 block = getBlock(sgbuf.num, dimention);
    dim3 grid = getGrid(block, sgbuf.num, dimention);

    void* sargs[] = { &sgbuf.axis, &sgbuf.num, &dimention, &seed };
    CUDA_ERROR_CHECK(cudaLaunchKernel(sphericalRandomKernel, grid, block, sargs, 0, NULL));

    int height = 1;
    float min = 1.f, max = 100.f;
    seed = rand();
    void* rargs[] = { &sgbuf.sharpness, &min, &max, &sgbuf.num, &height, &dimention, &seed };
    CUDA_ERROR_CHECK(cudaLaunchKernel(randomKernel, grid, block, rargs, 0, NULL));
}

void SGBuffer::normalize(SGBuffer& sgbuf) {
    int height = 1;
    dim3 block = getBlock(sgbuf.num, 1);
    dim3 grid = getGrid(block, sgbuf.num, 1);
    void* nargs[] = { &sgbuf.axis, &sgbuf.axis, &sgbuf.num, &height, &sgbuf.channel };
    CUDA_ERROR_CHECK(cudaLaunchKernel(normalizeKernel, grid, block, nargs, 0, NULL));

    float min = 0.f;
    float max = 8.f;
    int channel = 1;
    void* aargs[] = { &sgbuf.sharpness, &sgbuf.sharpness, &min ,&max, &sgbuf.num, &height, &channel };
    CUDA_ERROR_CHECK(cudaLaunchKernel(clampKernel, grid, block, aargs, 0, NULL));

    max = 1.f;
    block = getBlock(sgbuf.num, sgbuf.channel);
    grid = getGrid(block, sgbuf.num, sgbuf.channel);
    void* sargs[] = { &sgbuf.amplitude, &sgbuf.amplitude, &min ,&max, &sgbuf.num, &sgbuf.channel, &channel };
    CUDA_ERROR_CHECK(cudaLaunchKernel(clampKernel, grid, block, sargs, 0, NULL));

}

void SGBuffer::loadTXT(const char* path, SGBuffer* sgbuf) {
    FILE* file = fopen(path, "r");
    if (file == NULL) {
        printf("Impossible to open the file !\n");
        return;
    }
    SGBuffer::init(*sgbuf, 256, 3);
    vector<float> axis, sharpness, amplitude;
    for (int i = 0; i < sgbuf->num; i++) {
        float v[7];
        int matches = fscanf(file, "%f %f %f %f %f %f %f", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6]);
        if (matches != 7) {
            printf("Impossible to read the file !\n");
            return;
        }
        axis.push_back(v[0]);
        axis.push_back(v[1]);
        axis.push_back(v[2]);
        sharpness.push_back(v[3]);
        amplitude.push_back(v[4]);
        amplitude.push_back(v[5]);
        amplitude.push_back(v[6]);
    }
    CUDA_ERROR_CHECK(cudaMemcpy(sgbuf->axis, axis.data(), (size_t)sgbuf->num * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(sgbuf->sharpness, sharpness.data(), (size_t)sgbuf->num * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_ERROR_CHECK(cudaMemcpy(sgbuf->amplitude, amplitude.data(), (size_t)sgbuf->num * 3 * sizeof(float), cudaMemcpyHostToDevice));
}

__global__ void sgBakeKernel(const SGBuffer sgbuf, const Texture texture) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= texture.width || py >= texture.height)return;
    int pidx = px + texture.width * (py + texture.height * pz);

    float phi = (1.f - 2.f * (float)px / (float)texture.width) * 3.14159265f;
    float theta = (2.f * (float)py / (float)texture.height - 1.f) * 1.5707963f;
    float3 axis = make_float3(sin(phi) * cos(theta), sin(theta), cos(phi) * cos(theta));

    for (int i = 0; i < sgbuf.num; i++) {
        float3 sgaxis = ((float3*)sgbuf.axis)[i];
        float sgsharpness = sgbuf.sharpness[i];
        float sgvalue = exp(sgsharpness * (dot(axis, sgaxis) - 1.f));
        for (int k = 0; k < sgbuf.channel; k++) {
            AddNaNcheck(texture.texture[0][pidx * sgbuf.channel + k], sgbuf.amplitude[i * sgbuf.channel + k] * sgvalue);
        }
    }
}

void SGBuffer::bake(SGBuffer& sgbuf, Texture& texture) {
    Texture::liner(texture, 0.f, 0.f);
    dim3 block = getBlock(texture.width, texture.height);
    dim3 grid = getGrid(block, texture.width, texture.height);
    void* args[] = { &sgbuf, &texture };
    CUDA_ERROR_CHECK(cudaLaunchKernel(sgBakeKernel, grid, block, args, 0, NULL));
}

void SGBufferGrad::init(SGBufferGrad& sgbuf, int num, int channel) {
    SGBuffer::init(sgbuf, num, channel);
    CUDA_ERROR_CHECK(cudaMalloc(&sgbuf.axis, (size_t)num * 3 * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&sgbuf.sharpness, (size_t)num * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&sgbuf.amplitude, (size_t)num * channel * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMalloc(&sgbuf.buffer, (size_t)num * 3 * sizeof(float)));
}

__global__ void sgDisperseKernel(const SGBuffer sgbuf, float* buffer) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= sgbuf.num)return;
    float3 axis = ((float3*)sgbuf.axis)[px];
    float sharpness = sgbuf.sharpness[px];
    float3 force = make_float3(0.f, 0.f, 0.f);
    for (int i = 0; i < sgbuf.num; i++) {
        float sqDiffAmplitude = 0.f;
        for (int k = 0; k < sgbuf.channel; k++) {
            float diff = sgbuf.sharpness[px * sgbuf.channel + k] - sgbuf.sharpness[i * sgbuf.channel + k];
            sqDiffAmplitude += diff * diff;
        }
        float3 vec = axis - ((float3*)sgbuf.axis)[i];
        float l2 = max(dot(vec, vec), 1e-6);
        vec *= 1.f / sqrt(l2);
        force += vec * (1.f / max(l2 * sharpness * sqDiffAmplitude, 1e-6));
    }
    ((float3*)buffer)[px] = force;
}

__global__ void sgAddKernel(const SGBuffer sgbuf, float* buffer) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    if (px >= sgbuf.num)return;

    ((float3*)sgbuf.axis)[px] += ((float3*)buffer)[px];
}

void SGBufferGrad::disperse(SGBufferGrad& sgbuf) {
    dim3 block = getBlock(sgbuf.num, 1);
    dim3 grid = getGrid(block, sgbuf.num, 1);
    void* args[] = { &(SGBuffer)sgbuf, &sgbuf.buffer };
    CUDA_ERROR_CHECK(cudaLaunchKernel(sgDisperseKernel, grid, block, args, 0, NULL));
    CUDA_ERROR_CHECK(cudaLaunchKernel(sgAddKernel, grid, block, args, 0, NULL));
}

void SGBufferGrad::clear(SGBufferGrad& sgbuf) {
    CUDA_ERROR_CHECK(cudaMemset(sgbuf.axis, 0, (size_t)sgbuf.num * 3 * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemset(sgbuf.sharpness, 0, (size_t)sgbuf.num * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemset(sgbuf.amplitude, 0, (size_t)sgbuf.num * sgbuf.channel * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemset(sgbuf.buffer, 0, (size_t)sgbuf.num * 3 * sizeof(float)));
}


void GLbuffer::init(GLbuffer& rb, float* buffer, int width, int height, int channel) {
    rb.width = width;
    rb.height = height;
    rb.channel = channel;
    rb.buffer = buffer;
    CUDA_ERROR_CHECK(cudaMallocHost(&rb.gl_buffer, rb.Size()));
    glGenTextures(1, &rb.id);
    glBindTexture(GL_TEXTURE_2D, rb.id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT15, rb.id, 0);
}

void GLbuffer::draw(GLbuffer& rb, GLint internalformat, GLenum format, float minX, float minY, float maxX, float maxY) {
    CUDA_ERROR_CHECK(cudaMemcpy(rb.gl_buffer, rb.buffer, rb.Size(), cudaMemcpyDeviceToHost));
    glBindTexture(GL_TEXTURE_2D, rb.id);
    glTexImage2D(GL_TEXTURE_2D, 0, internalformat, rb.width, rb.height, 0, format, GL_FLOAT, rb.gl_buffer);
    glBegin(GL_POLYGON);
    glTexCoord2f(0.f, 0.f); glVertex2f(minX, minY);
    glTexCoord2f(0.f, 1.f); glVertex2f(minX, maxY);
    glTexCoord2f(1.f, 1.f); glVertex2f(maxX, maxY);
    glTexCoord2f(1.f, 0.f); glVertex2f(maxX, minY);
    glEnd();
}

void GLbuffer::draw(GLbuffer& rb, GLint internalformat, GLenum format, float texminX, float texminY, float texmaxX, float texmaxY, float minX, float minY, float maxX, float maxY) {
    CUDA_ERROR_CHECK(cudaMemcpy(rb.gl_buffer, rb.buffer, rb.Size(), cudaMemcpyDeviceToHost));
    glBindTexture(GL_TEXTURE_2D, rb.id);
    glTexImage2D(GL_TEXTURE_2D, 0, internalformat, rb.width, rb.height, 0, format, GL_FLOAT, rb.gl_buffer);
    glBegin(GL_POLYGON);
    glTexCoord2f(texminX, texminY); glVertex2f(minX, minY);
    glTexCoord2f(texminX, texmaxY); glVertex2f(minX, maxY);
    glTexCoord2f(texmaxX, texmaxY); glVertex2f(maxX, maxY);
    glTexCoord2f(texmaxX, texminY); glVertex2f(maxX, minY);
    glEnd();
}