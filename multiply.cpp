#include <iostream>
#include <vector_types.h>
#include <chrono>
#include "types.h"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include <iomanip>
#include "random.h"
#include "timeMeasure.h"

const std::string messagePixel = " pixels ";

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

template<typename T> void launchCudaProcess0(dim3 grid, dim3 block,
					T *gain, T *imageInput, T *imageOutput, int imgW);

template<typename T> void launchCudaProcess1(dim3 grid, dim3 block,
					T *gain, T *imageInput, T *imageOutput, int imgW);

template<> void launchCudaProcess0<float>(dim3 grid, dim3 block,
					float *gain, float *imageInput, float *imageOutput, int imgW)
{
	launchCudaProcessFloat0(grid, block, gain, imageInput, imageOutput, imgW);
}

template<> void launchCudaProcess0<short>(dim3 grid, dim3 block,
					short *gain, short *imageInput, short *imageOutput, int imgW)
{
	launchCudaProcessHalf0(grid, block, gain, imageInput, imageOutput, imgW);
}

template<> void launchCudaProcess1<float>(dim3 grid, dim3 block,
					float *gain, float *imageInput, float *imageOutput, int imgW)
{
	//launchCudaProcessFloat1(grid, block, gain, imageInput, imageOutput, imgW);
}

template<> void launchCudaProcess1<short>(dim3 grid, dim3 block,
					short *gain, short *imageInput, short *imageOutput, int imgW)
{
	launchCudaProcessHalf1(grid, block, gain, imageInput, imageOutput, imgW);
}

template<typename T> void launchCudaProcess(int imgW, int imgH, int gridX, int gridY, int cLoop, enum processType t = elementWise)
{
	T* srcImage, *gainImage, *dstImage, *cpuSrc, *cpuGain;
	int s = imgW*imgH;

	cudaMalloc((T**)&srcImage, (s*sizeof(T)));
	cudaMalloc((T**)&gainImage, (s*sizeof(T)));
	cudaMalloc((T**)&dstImage, (s*sizeof(T)));
	cpuSrc = (T*)malloc(s*sizeof(T));
	cpuGain = (T*)malloc(s*sizeof(T));

	fillRandomNumber<T>(cpuSrc, s);
	fillRandomNumber<T>(cpuGain, s);

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
	launchCudaProcess0<T>(grid, block, gainImage, srcImage, dstImage, imgW);

	std::vector<duration> timeDuration1;
	std::vector<duration> timeDuration2;
	for(int i = 0;i < cLoop;i++)
	{
		auto start = std::chrono::system_clock::now();
		cudaMemcpy(srcImage, cpuSrc, s*sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(gainImage, cpuGain, s*sizeof(T), cudaMemcpyHostToDevice);
		auto middle = std::chrono::system_clock::now();
		duration usec1 = std::chrono::duration_cast<timeResolution>(middle - start).count();
		duration usec2 = 0;
		switch(t)
		{
			case pack2:
				{
					auto mid1 = std::chrono::system_clock::now();
					launchCudaProcess1<T>(grid, block, gainImage, srcImage, dstImage, imgW);
					auto end  = std::chrono::system_clock::now();
					usec2 = std::chrono::duration_cast<timeResolution>(end - mid1).count();
				}
				break;
			default:
				{
					auto mid1 = std::chrono::system_clock::now();
					launchCudaProcess0<T>(grid, block, gainImage, srcImage, dstImage, imgW);
					auto end  = std::chrono::system_clock::now();
					usec2 = std::chrono::duration_cast<timeResolution>(end - mid1).count();
				}
				break;
		}
		timeDuration1.push_back(usec1);
		timeDuration2.push_back(usec2);
	}
	std::cout << extractDuration(timeDuration1) << " + " << extractDuration(timeDuration2) << messageTime << s << messagePixel << '(' << imgW << 'x' << imgH << ')' << std::endl;

	cudaFree((void*)srcImage);
	cudaFree((void*)gainImage);
	cudaFree((void*)dstImage);
	free((void*)cpuSrc);
	free((void*)cpuGain);
}

int main()
{
	launchCudaProcess<float>(1920, 1080, 16, 16, 1);
	launchCudaProcess<short>(1920, 1080, 16, 16, 1);
	launchCudaProcess<short>(1920, 1080, 16, 16, 1, pack2);
	launchCudaProcess<float>(640,  480,  16, 16, 100);
	launchCudaProcess<short>(640,  480,  16, 16, 100);
	launchCudaProcess<short>(640,  480,  16, 16, 100, pack2);
	launchCudaProcess<float>(1920, 1080, 16, 16, 100);
	launchCudaProcess<short>(1920, 1080, 16, 16, 100);
	launchCudaProcess<short>(1920, 1080, 16, 16, 100, pack2);
	launchCudaProcess<float>(3840, 2160, 16, 16, 100);
	launchCudaProcess<short>(3840, 2160, 16, 16, 100);
	launchCudaProcess<short>(3840, 2160, 16, 16, 100, pack2);
	launchCudaProcess<float>(4000, 4000, 16, 16, 100);
	launchCudaProcess<short>(4000, 4000, 16, 16, 100);
	launchCudaProcess<short>(4000, 4000, 16, 16, 100, pack2);
	launchCudaProcess<float>(8000, 8000, 16, 16, 100);
	launchCudaProcess<short>(8000, 8000, 16, 16, 100);
	launchCudaProcess<short>(8000, 8000, 16, 16, 100, pack2);

	return 0;
}
