#include "common.h"

dim3 getBlock(int width, int height) {
    if (width  >= MAX_DIM_PER_BLOCK &&  height >= MAX_DIM_PER_BLOCK) {
        return dim3(MAX_DIM_PER_BLOCK, MAX_DIM_PER_BLOCK);
    }
    else if (width >= MAX_DIM_PER_BLOCK) {
        int w = MAX_DIM_PER_BLOCK * MAX_DIM_PER_BLOCK, h = 1;
        while (h < height) {
            w >>= 1;
            h <<= 1;
        }
        return dim3(w, h);
    }
    else if (height >= MAX_DIM_PER_BLOCK) {
        int w = 1, h =MAX_DIM_PER_BLOCK * MAX_DIM_PER_BLOCK ;
        while (w < width) {
            w <<= 1;
            h >>= 1;
        }
        return dim3(w, h);
    }
    else {
        int w = 1, h = 1;
        while (w < width)w <<= 1;
        while (h < height)h <<= 1;
        return dim3(w, h);
    }
}

dim3 getGrid(dim3 block, int width, int height) {
    dim3 grid;
    grid.x = (width + block.x - 1) / block.x;
    grid.y = (height + block.y - 1) / block.y;
    return grid;
}

dim3 getGrid(dim3 block, int width, int height, int depth) {
    dim3 grid;
    grid.x = (width + block.x - 1) / block.x;
    grid.y = (height + block.y - 1) / block.y;
    grid.z = (depth + block.z - 1) / block.z;
    return grid;
}

int MSB(size_t v) {
    v |= (v >> 1);
    v |= (v >> 2);
    v |= (v >> 4);
    v |= (v >> 8);
    v |= (v >> 16);
    v |= (v >> 32);
    v ^= (v >> 1);
    int msb = 0;
    if (v & 0xffffffff00000000)msb += 32;
    if (v & 0xffff0000ffff0000)msb += 16;
    if (v & 0xff00ff00ff00ff00)msb += 8;
    if (v & 0xf0f0f0f0f0f0f0f0)msb += 4;
    if (v & 0xcccccccccccccccc)msb += 2;
    if (v & 0xaaaaaaaaaaaaaaaa)msb += 1;
    return msb;
}

int LSB(size_t v) {
    v |= (v << 1);
    v |= (v << 2);
    v |= (v << 4);
    v |= (v << 8);
    v |= (v << 16);
    v |= (v << 32);
    v ^= (v << 1);
    int msb = 0;
    if (v & 0xffffffff00000000)msb += 32;
    if (v & 0xffff0000ffff0000)msb += 16;
    if (v & 0xff00ff00ff00ff00)msb += 8;
    if (v & 0xf0f0f0f0f0f0f0f0)msb += 4;
    if (v & 0xcccccccccccccccc)msb += 2;
    if (v & 0xaaaaaaaaaaaaaaaa)msb += 1;
    return msb;
}