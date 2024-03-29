#pragma once
#include "common.h"

#define TEX_MAX_MIP_LEVEL 16

struct Buffer {
	float* buffer;
	int num;
	int dimention;
	size_t Size() { return (size_t)num * dimention * sizeof(float); };
	static void init(Buffer& buf, int num, int dimention);
	static void init(Buffer& buf, Buffer& src, int dimention);
	static void copy(Buffer& dst, Buffer& src);
	static void copy(Buffer& dst, float* src);
	static void copy(float* dst, Buffer& src);
	static void liner(Buffer& buf, float w, float b);
	static void addRandom(Buffer& buf, float min, float max);
	static void clamp(Buffer& buf, float min, float max);
	static void normalize(Buffer& buf);
};

struct BufferGrad :Buffer {
	float* grad;
	static void init(BufferGrad& buf, int num, int dimention);
	static void init(BufferGrad& buf, Buffer& src, int dimention);
	static void clear(BufferGrad& buf);
};

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
	static void liner(Attribute& attr, float w, float b);
	static float distanceError(Attribute& predict, Attribute& target);
	static void addRandom(Attribute& attr, float min, float max);
	static void step(Attribute& attr, float threshold);
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
	static void init(Texture& texture, float* original, int width, int height, int channel, int miplevel);
	static void buildMIP(Texture& texture);	
	static void bilinearDownsampling(Texture& texture);
	static void loadBMP(const char* path, Texture& texture, int miplevel);
	static void setColor(Texture& texture, float* color);
	static void liner(Texture& texture, float w, float b);
	static void normalize(Texture& texture);
	static void addRandom(Texture& texture, float min, float max);
	static void clamp(Texture& texture, float min, float max);
};

struct TextureGrad : Texture {
	float* grad[TEX_MAX_MIP_LEVEL];
	static void init(TextureGrad& texture, int width, int height, int channel, int miplevel);
	static void clear(TextureGrad& texture);
	static void gradSumup(TextureGrad& texture);
};



struct SGBuffer {
	float* axis;
	float* sharpness;
	float* amplitude;
	int num;
	int channel;
	static void init(SGBuffer& sgbuf, int num, int channel);
	static void copy(SGBuffer& dst, float* axis, float* sharpness, float* amplitude);
	static void randomize(SGBuffer& sgbuf);
	static void normalize(SGBuffer& sgbuf);
	static void loadTXT(const char* path, SGBuffer* sgbuf);
	static void bake(SGBuffer& sgbuf, Texture& texture);
};

struct SGBufferGrad :SGBuffer {
	float* axis;
	float* sharpness;
	float* amplitude;
	float* buffer;
	static void init(SGBufferGrad& sgbuf, int num, int channel);
	static void disperse(SGBufferGrad& sgbuf);
	static void clear(SGBufferGrad& sgbuf);
};



struct GLbuffer {
	GLuint id;
	float* gl_buffer;
	float* buffer;
	int width;
	int height;
	int channel;
	size_t Size() { return (size_t)width * height * channel * sizeof(float); };
	static void init(GLbuffer& rb, float* buffer, int width, int height, int channel);
	static void draw(GLbuffer& rb, GLint internalformat, GLenum format, float minX, float minY, float maxX, float maxY);
	static void draw(GLbuffer& rb, GLint internalformat, GLenum format, float texminX, float texminY, float texmaxX, float texmaxY, float minX, float minY, float maxX, float maxY);
};