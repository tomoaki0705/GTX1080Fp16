#include <iostream>
#include <vector_types.h>
#include <chrono>
#include "types.h"
#include <cuda_runtime.h>

extern "C" void
launchCudaProcessHalf0(dim3 grid, dim3 block, int sbytes,
					short *gain, short *imageInput, short *imageOutput, int imgW);

extern "C" void
launchCudaProcessFloat0(dim3 grid, dim3 block, int sbytes,
					float *gain, float *imageInput, float *imageOutput, int imgW);


template<typename T> void launchCudaProcess(int imgW, int imgH, int gridX, int gridY);

template<> void
launchCudaProcess<float>(int imgW, int imgH, int gridX, int gridY)
{
	float* imageFloat, *gainFloat, *dstFloat;
	int s = imgW*imgH;

	cudaMalloc((float**)&imageFloat, (s*sizeof(float)));
	cudaMalloc((float**)&gainFloat, (s*sizeof(float)));
	cudaMalloc((float**)&dstFloat, (s*sizeof(float)));

	dim3 block(gridX, gridY, 1);
	dim3 grid(imgW / block.x, imgH / block.y, 1);

	auto start = std::chrono::system_clock::now();
	launchCudaProcessFloat0(grid, block, 0, gainFloat, imageFloat, dstFloat, imgW);
	auto end  = std::chrono::system_clock::now();
	auto dur = end - start;
	int msec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
	std::cout << msec << " [us] consumed" << std::endl;

	cudaFree((void*)imageFloat);
	cudaFree((void*)gainFloat);
	cudaFree((void*)dstFloat);
}


int main()
{

	launchCudaProcess<float>(1920, 1080, 16, 16);

	return 0;
}
