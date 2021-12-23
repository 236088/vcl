#include "preset.h"

void Noise::init(NoiseParams& np, RasterizeParams& rast, float* in, int channel, float intensity) {
	np.kernel.width = rast.kernel.width;
	np.kernel.height = rast.kernel.height;
	np.kernel.depth = rast.kernel.depth;
	np.kernel.channel = channel;
	np.kernel.intensity = intensity;
	np.kernel.in = in;
	CUDA_ERROR_CHECK(cudaMalloc(&np.kernel.out, np.Size()));
}

__global__ void NoiseForwardKernel(const NoiseKernelParams np,unsigned int seed) {
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
	int pz = blockIdx.z;
	if (px >= np.width || py >= np.height || pz >= np.depth)return;
	int pidx = px + np.width * (py + np.height * pz);
	for (int i = 0; i < np.channel; i++) {
		np.out[pidx * np.channel + i] = np.in[pidx * np.channel + i] + np.intensity * (getUniform(pidx, seed + i, 0xcafef00d) - np.in[pidx * np.channel + i]);
	}
}

void Noise::forward(NoiseParams& np) {
	CUDA_ERROR_CHECK(cudaMemset(np.kernel.out, 0, np.Size()));
	unsigned int seed = rand();
	dim3 block = getBlock(np.kernel.width, np.kernel.height);
	dim3  grid = getGrid(block, np.kernel.width, np.kernel.height);
	void* args[] = { &np.kernel ,&seed };
	CUDA_ERROR_CHECK(cudaLaunchKernel(NoiseForwardKernel, grid, block, args, 0, NULL));
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


