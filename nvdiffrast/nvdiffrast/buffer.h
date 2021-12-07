#pragma once
#include "common.h"

#define TEX_MAX_MIP_LEVEL 16

struct Attribute {
	float* vbo;
	unsigned int* vao;
	int vboNum;
	int vaoNum;
	int dimention;
	size_t vboSize() { return (size_t)vboNum * dimention * sizeof(float); };
	size_t vaoSize() { return (size_t)vaoNum * 3 * sizeof(unsigned int); };
	static void init(Attribute& attr, int vboNum, int vaoNum, int dimention);
	static void init(Attribute& attr, Attribute& src, int dimention);
	static void loadOBJ(const char* path, Attribute* pos, Attribute* texel, Attribute* normal); 
	static void copy(Attribute& dst, Attribute& src);
	static void affine(Attribute& attr, float w, float b);
	static void addRandom(Attribute& attr, float min, float max);
};

struct AttributeGrad :Attribute {
	float* grad;
	static void init(AttributeGrad& attr, int vboNum, int vaoNum, int dimention);
	static void init(AttributeGrad& attr, Attribute& src, int dimention);
	static void clear(AttributeGrad& attr);
};

struct Texture {
	int width;
	int height;
	int channel;
	int miplevel;
	float* texture[TEX_MAX_MIP_LEVEL];
	size_t Size() { return (size_t)width * height * channel * sizeof(float); };
	static void init(Texture& texture, int width, int height, int channel, int miplevel);
	static void buildMIP(Texture& texture);
	static void loadBMP(const char* path, Texture& texture, int miplevel);
};

struct TextureGrad : Texture {
	float* grad[TEX_MAX_MIP_LEVEL];
	static void init(TextureGrad& texture, int width, int height, int channel, int miplevel);
	static void clear(TextureGrad& texture);
	static void gradSumup(TextureGrad& texture);
};