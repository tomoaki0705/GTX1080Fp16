#include <iostream>
#include <vector_types.h>
#include <chrono>
#include "types.h"
#include <cuda_runtime.h>

extern "C" void
launchCudaProcessHalf0(dim3 grid, dim3 block,
					short *gain, short *imageInput, short *imageOutput, int imgW);

extern "C" void
launchCudaProcessFloat0(dim3 grid, dim3 block,
					float *gain, float *imageInput, float *imageOutput, int imgW);


template<typename T> void launchCudaProcess(int imgW, int imgH, int gridX, int gridY, int cLoop);

template<> void
launchCudaProcess<float>(int imgW, int imgH, int gridX, int gridY, int cLoop)
{
	float* srcImage, *gainImage, *dstImage;
	int s = imgW*imgH;

	cudaMalloc((float**)&srcImage, (s*sizeof(float)));
	cudaMalloc((float**)&gainImage, (s*sizeof(float)));
	cudaMalloc((float**)&dstImage, (s*sizeof(float)));

	dim3 block(gridX, gridY, 1);
	dim3 grid(imgW / block.x, imgH / block.y, 1);

	auto start = std::chrono::system_clock::now();
	for(int i = 0;i < cLoop;i++)
	{
		launchCudaProcessFloat0(grid, block, gainImage, srcImage, dstImage, imgW);
	}
	auto end  = std::chrono::system_clock::now();
	auto dur = end - start;
	int msec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
	std::cout << msec << " [us] consumed" << std::endl;

	cudaFree((void*)srcImage);
	cudaFree((void*)gainImage);
	cudaFree((void*)dstImage);
}

template<> void
launchCudaProcess<short>(int imgW, int imgH, int gridX, int gridY, int cLoop)
{
	short* srcImage, *gainImage, *dstImage;
	int s = imgW*imgH;

	cudaMalloc((short**)&srcImage, (s*sizeof(short)));
	cudaMalloc((short**)&gainImage, (s*sizeof(short)));
	cudaMalloc((short**)&dstImage, (s*sizeof(short)));

	dim3 block(gridX, gridY, 1);
	dim3 grid(imgW / block.x, imgH / block.y, 1);

	auto start = std::chrono::system_clock::now();
	for(int i = 0;i < cLoop;i++)
	{
		launchCudaProcessHalf0(grid, block, gainImage, srcImage, dstImage, imgW);
	}
	auto end  = std::chrono::system_clock::now();
	auto dur = end - start;
	int msec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
	std::cout << msec << " [us] consumed" << std::endl;

	cudaFree((void*)srcImage);
	cudaFree((void*)gainImage);
	cudaFree((void*)dstImage);
}


int main()
{

	launchCudaProcess<float>(1920, 1080, 16, 16, 1000);
	launchCudaProcess<short>(1920, 1080, 16, 16, 1000);
	launchCudaProcess<float>(640,  480,  16, 16, 1000);
	launchCudaProcess<short>(640,  480,  16, 16, 1000);

	return 0;
}
