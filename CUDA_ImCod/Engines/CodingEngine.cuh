#pragma once
#ifndef CODINGENGINE_CUH
#define CODINGENGINE_CUH
#include "Engine.cuh"

class CodingEngine : public Engine
{

	void engineManager(int cType);
	void initMemory(bool typeOfCoding);
	
	void runImage();
	void readGrayScaleImage();
	void readRGBImage();
	void prepareRGBImage(cudaStream_t mainStream);
	void writeRGBBitStream(cudaStream_t mainStream, int iter);
	void writeCodedBitStream();
	void Bridge(cudaStream_t mainStream, int* input, int* output);
	void prepareGrayScaleImage(cudaStream_t mainStream);
};
__device__ float const _irreversibleColorTransformForward[3][3] = { { 0.299f, 0.587f, 0.114f },{ -0.168736f, -0.331264f, 0.5f },{ 0.5f, -0.418688f, -0.081312f } };
__global__ void RGBTransformLossless(unsigned char* inputR, unsigned char* inputG, unsigned char* inputB, int* outputR, int* outputG, int* outputB, int bitdepth, bool uSigned);

__global__ void RGBTransformLossy(unsigned char* inputR, unsigned char* inputG, unsigned char* inputB, float* outputR, float* outputG, float* outputB, int bitdepth, bool uSigned);


__global__ void expandShortToIntKernel(unsigned short* input, int* output, int size);
__global__ void KernelBridge(unsigned int* input, int* output, int size);
#endif