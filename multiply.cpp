#include <iostream>
#include <vector_types.h>
#include <chrono>
#include "types.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

extern "C" void
launchCudaProcessHalf0(dim3 grid, dim3 block,
					short *gain, short *imageInput, short *imageOutput, int imgW);

extern "C" void
launchCudaProcessFloat0(dim3 grid, dim3 block,
					float *gain, float *imageInput, float *imageOutput, int imgW);

extern "C" void
launchCudaProcessHalf1(dim3 grid, dim3 block,
					short *gain, short *imageInput, short *imageOutput, int imgW);

extern "C" void
launchCudaProcessFloat1(dim3 grid, dim3 block,
					float *gain, float *imageInput, float *imageOutput, int imgW);

int getMedian(std::vector<int>& timeDuration)
{
	std::sort(timeDuration.begin(), timeDuration.end());
	return timeDuration[timeDuration.size()/2];
}

template<typename T> void launchCudaProcess(int imgW, int imgH, int gridX, int gridY, int cLoop, enum processType t = elementWise);

template<> void
launchCudaProcess<float>(int imgW, int imgH, int gridX, int gridY, int cLoop, enum processType t)
{
	float* srcImage, *gainImage, *dstImage;
	int s = imgW*imgH;

	cudaMalloc((float**)&srcImage, (s*sizeof(float)));
	cudaMalloc((float**)&gainImage, (s*sizeof(float)));
	cudaMalloc((float**)&dstImage, (s*sizeof(float)));

	dim3 block(gridX, gridY, 1);
	dim3 grid(imgW / block.x, imgH / block.y, 1);
	switch(t)
	{
		case pack2:
			grid = dim3((imgW / block.x) / 2, imgH / block.y, 1);
			break;
		case pack4:
		case elementWise:
		default:
			grid = dim3(imgW / block.x, imgH / block.y, 1);
			break;
	}

	std::vector<int> timeDuration;
	for(int i = 0;i < cLoop;i++)
	{
		auto start = std::chrono::system_clock::now();
		launchCudaProcessFloat0(grid, block, gainImage, srcImage, dstImage, imgW);
		auto end  = std::chrono::system_clock::now();
		auto dur = end - start;
		int usec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
		timeDuration.push_back(usec);
	}
	std::cout << getMedian(timeDuration) << " [us] consumed" << std::endl;

	cudaFree((void*)srcImage);
	cudaFree((void*)gainImage);
	cudaFree((void*)dstImage);
}

template<> void
launchCudaProcess<short>(int imgW, int imgH, int gridX, int gridY, int cLoop, enum processType t)
{
	short* srcImage, *gainImage, *dstImage;
	int s = imgW*imgH;

	cudaMalloc((short**)&srcImage, (s*sizeof(short)));
	cudaMalloc((short**)&gainImage, (s*sizeof(short)));
	cudaMalloc((short**)&dstImage, (s*sizeof(short)));

	dim3 block(gridX, gridY, 1);
	dim3 grid(imgW / block.x, imgH / block.y, 1);
	switch(t)
	{
		case pack2:
			grid = dim3((imgW / block.x) / 2, imgH / block.y, 1);
			break;
		case pack4:
		case elementWise:
		default:
			grid = dim3(imgW / block.x, imgH / block.y, 1);
			break;
	}


	std::vector<int> timeDuration;
	switch(t)
	{
		case pack2:
			for(int i = 0;i < cLoop;i++)
			{
				auto start = std::chrono::system_clock::now();
				launchCudaProcessHalf1(grid, block, gainImage, srcImage, dstImage, imgW);
				auto end  = std::chrono::system_clock::now();
				auto dur = end - start;
				int usec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
				timeDuration.push_back(usec);
			}
			break;
		default:
			for(int i = 0;i < cLoop;i++)
			{
				auto start = std::chrono::system_clock::now();
				launchCudaProcessHalf0(grid, block, gainImage, srcImage, dstImage, imgW);
				auto end  = std::chrono::system_clock::now();
				auto dur = end - start;
				int usec = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
				timeDuration.push_back(usec);
			}
			break;
	}
	std::cout << getMedian(timeDuration) << " [us] consumed" << std::endl;

	cudaFree((void*)srcImage);
	cudaFree((void*)gainImage);
	cudaFree((void*)dstImage);
}


int main()
{

	launchCudaProcess<float>(1920, 1080, 16, 16, 1000);
	launchCudaProcess<short>(1920, 1080, 16, 16, 1000);
	launchCudaProcess<short>(1920, 1080, 16, 16, 1000, pack2);
	launchCudaProcess<float>(640,  480,  16, 16, 1000);
	launchCudaProcess<short>(640,  480,  16, 16, 1000);
	launchCudaProcess<short>(640,  480,  16, 16, 1000, pack2);
	launchCudaProcess<float>(3840, 2160, 16, 16, 1000);
	launchCudaProcess<short>(3840, 2160, 16, 16, 1000);
	launchCudaProcess<short>(3840, 2160, 16, 16, 1000, pack2);

	return 0;
}
