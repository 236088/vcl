#include "buffer.h"

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

    std::vector<float> tempPos, tempTexel, tempNormal;
    std::vector<unsigned int> tempPosIndex, tempTexelIndex, tempNormalIndex;
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

__global__ void AttributeLinerKernel(float* attr, float w, float b, int width, int height) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height)return;
    int pidx = px + width * py;
    attr[pidx] = attr[pidx] * w + b;
}

void Attribute::liner(Attribute& attr, float w, float b) {
    dim3 block = getBlock(attr.vboNum, attr.dimention);
    dim3 grid = getGrid(block, attr.vboNum, attr.dimention);
    void* args[] = { &attr.vbo, &w,&b,&attr.vboNum,&attr.dimention };
    CUDA_ERROR_CHECK(cudaLaunchKernel(AttributeLinerKernel, grid, block, args, 0, NULL));
}

__global__ void AttributeRandomKernel(float* buffer, float min, float max, int width, int height, unsigned int seed) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height)return;
    int pidx = px + width * py;
    buffer[pidx] += min + (max - min) * getUniform(pidx, seed, 0xba5ec0de);
}

void Attribute::addRandom(Attribute& attr, float min, float max) {
    unsigned int seed = rand();
    dim3 block = getBlock(attr.vboNum, attr.dimention);
    dim3 grid = getGrid(block, attr.vboNum, attr.dimention);
    void* args[] = { &attr.vbo,&min,&max,&attr.vboNum,&attr.dimention, &seed };
    CUDA_ERROR_CHECK(cudaLaunchKernel(AttributeRandomKernel, grid, block, args, 0, NULL));
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

__global__ void TextureSetColorKernel (const Texture texture, int index, int width, int height, float* color) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= width || py >= height)return;
    int pidx = px + width * (py + height * pz);

    for (int i = 0; i < texture.channel; i++) {
        texture.texture[index][pidx * texture.channel + i] = color[i];
    }
}

void Texture::setColor(Texture& texture, float* color) {
    int w = texture.width, h = texture.height;
    int i = 0;
    float* dev_color;
    CUDA_ERROR_CHECK(cudaMalloc(&dev_color, (size_t)texture.channel * sizeof(float)));
    CUDA_ERROR_CHECK(cudaMemcpy(dev_color, color, (size_t)texture.channel * sizeof(float), cudaMemcpyHostToDevice));
    void* args[] = { &texture,&i, &w, &h ,&dev_color };
    for (; i < texture.miplevel; i++) {
        dim3 block = getBlock(w, h);
        dim3 grid = getGrid(block, w, h);
        CUDA_ERROR_CHECK(cudaLaunchKernel(TextureSetColorKernel, grid, block, args, 0, NULL));
        w >>= 1; h >>= 1;
    }
    CUDA_ERROR_CHECK(cudaFree(dev_color));
}

__global__ void TextureLinerKernel (const Texture texture, int index, int width, int height, float w, float b) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= width || py >= height)return;
    int pidx = px + width * (py + height * pz);

    for (int i = 0; i < texture.channel; i++) {
        texture.texture[index][pidx * texture.channel + i] = texture.texture[index][pidx * texture.channel + i] * w + b;
    }
}

void Texture::liner(Texture& texture, float w, float b) {
    int w_ = texture.width, h_ = texture.height;
    int i = 0;
    void* args[] = { &texture,&i, &w_, &h_, &w, &b};
    for (; i < texture.miplevel; i++) {
        dim3 block = getBlock(w_, h_);
        dim3 grid = getGrid(block, w_, h_);
        CUDA_ERROR_CHECK(cudaLaunchKernel(TextureLinerKernel, grid, block, args, 0, NULL));
        w_ >>= 1; h_ >>= 1;
    }
}

__global__ void normalizeKernel (const Texture texture, int index, int width, int height) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= width || py >= height)return;
    int pidx = px + width * (py + height * pz);

    float s = 0.f;
    for (int i = 0; i < texture.channel; i++) {
        float v = texture.texture[index][pidx * texture.channel + i];
        s += v * v;
    }
    s = 1.f/sqrt(s);
    if (isfinite(s)) {
        for (int i = 0; i < texture.channel; i++) {
            texture.texture[index][pidx * texture.channel + i] *= s;
        }
    }
    else {
        for (int i = 0; i < texture.channel - 1; i++) {
            texture.texture[index][pidx * texture.channel + i] = 0.f;
        }
        texture.texture[index][(pidx + 1) * texture.channel - 1] = 1.f;
    }
}

void Texture::normalize(Texture& texture) {
    int w = texture.width, h = texture.height;
    int i = 0;
    void* args[] = { &texture,&i, &w, &h};
    for (; i < texture.miplevel; i++) {
        dim3 block = getBlock(w, h);
        dim3 grid = getGrid(block, w, h);
        CUDA_ERROR_CHECK(cudaLaunchKernel(normalizeKernel, grid, block, args, 0, NULL));
        w >>= 1; h >>= 1;
    }
}

__global__ void TextureRandomKernel(const Texture texture, float max, float min, unsigned int seed) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= texture.width || py >= texture.height)return;
    int pidx = px + texture.width * (py + texture.height * pz);

    for (int i = 0; i < texture.channel; i++) {
        texture.texture[0][pidx * texture.channel + i] = min + (max - min) * getUniform(pidx * texture.channel + i,seed,0xf122ba22);
    }
}

void Texture::addRandom(Texture& texture, float max, float min) {
    int w_ = texture.width, h_ = texture.height;
    unsigned int seed = rand();
    dim3 block = getBlock(texture.width, texture.height);
    dim3 grid = getGrid(block, texture.width, texture.height);
    void* args[] = { &texture,&max,&min ,&seed };
    CUDA_ERROR_CHECK(cudaLaunchKernel(TextureRandomKernel, grid, block, args, 0, NULL));
    buildMIP(texture);
}

__global__ void TextureClampKernel(const Texture texture, float min, float max) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int pz = blockIdx.z;
    if (px >= texture.width || py >= texture.height)return;
    int pidx = px + texture.width * (py + texture.height * pz);

    for (int i = 0; i < texture.channel; i++) {
        texture.texture[0][pidx * texture.channel + i] = clamp(texture.texture[0][pidx * texture.channel + i], min, max);
    }
}

void Texture::clamp(Texture& texture, float min, float max) {
    int w_ = texture.width, h_ = texture.height;
    dim3 block = getBlock(texture.width, texture.height);
    dim3 grid = getGrid(block, texture.width, texture.height);
    void* args[] = { &texture,&min ,&max};
    CUDA_ERROR_CHECK(cudaLaunchKernel(TextureClampKernel, grid, block, args, 0, NULL));
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