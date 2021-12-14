#pragma once
#include "common.h"
#include "buffer.h"

struct OptimizeKernelParams {
	float* param;
	float* grad;

	int size;
	int width;
	int height;
	int depth;
};

struct OptimizeParams {
	OptimizeKernelParams kernel;
	int it;

	dim3 block;
	dim3 grid;
	size_t Size() { return (size_t)kernel.size * sizeof(float); };
};

class Optimizer {
public:
	static void init(OptimizeParams& opt, float* param, float* grad, int size, int width, int height, int depth);
	static void init(OptimizeParams& opt, AttributeGrad& attr);
	static void init(OptimizeParams& opt, TextureGrad& texture);
	static void randomParams(OptimizeParams& opt, float min, float max);
	static void clampParams(OptimizeParams& opt, float min, float max);
};



struct AdamHyperParams {
	double rhom;
	double rhov;
	double rhomt;
	double rhovt;
	double eta;
	double eps;
	float* m;
	float* v;
};

struct AdamParams : OptimizeParams {
	AdamHyperParams hyper;
};

class Adam : Optimizer {
public:
	static void setHyperParams(AdamParams& adam,double eta,  double rhom, double rhov, double eps);
	static void step(AdamParams& adam);
};

struct NadamHyperParams{
	double mupow;
	double mupowt;
	double mu;
	double mu0;
	double mu1;
	double rho;
	double alpha;
	double mut0;
	double mut1;
	double rhot;
	double eps;
	float* m;
	float* v;
};

struct NadamParams : OptimizeParams {
	NadamHyperParams hyper;
};


class Nadam : Optimizer {
public:
	static void init(NadamParams& nadam, float* param, float* grad, int size, int width, int height, int depth);
	static void init(NadamParams& nadam, Attribute& attr, float* grad);
	static void setHyperParams(NadamParams& nadam, double alpha, double mu, double rho, double eps);
	static void step(NadamParams& adam);
};
