#include <iostream>
#include <vector_types.h>
#include <chrono>
#include "types.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <iomanip>


const std::string messageTime = " [ns] elapsed\t";
const std::string messagePixel = " pixels ";
#define timeResolution std::chrono::nanoseconds
typedef int duration;

const uint64_t seed = 0x1234567;
uint64_t state = seed;

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

duration getMedian(std::vector<duration>& timeDuration)
{
	std::sort(timeDuration.begin(), timeDuration.end());
	return timeDuration[timeDuration.size()/2];
}

inline unsigned RNG()
{
	state = (uint64_t)(unsigned)state* 4164903690U + (unsigned)(state >> 32);
	return (unsigned)state;
}

template<typename T> void fillRandomNumber(T* array, int cElement);

template<> void fillRandomNumber<float>(float* array, int cElement)
{
	for(unsigned int i = 0;i < cElement;i++)
	{
		short random = (short)(RNG() & 0x3fff);
		float exp = (float)(random - 0x2000) / (float)(0x1000);
		array[i] = (float)pow(2, exp);
	}
}

template<> void fillRandomNumber<short>(short* array, int cElement)
{
	for(unsigned int i = 0;i < cElement;i++)
	{
		short random = (short)(RNG() & 0x3fff);
		float exp = (float)(random - 0x2000) / (float)(0x1000);
		float var[4] = {(float)pow(2, exp), 0, 0, 0,};
		__m128 vF = _mm_load_ps(var);
		__m128i vH = _mm_cvtps_ph(vF, 0);
		short half[4];
		_mm_storel_epi64((__m128i*)half, vH);
		array[i] = half[0];
	}
}

template<typename T> void launchCudaProcess(int imgW, int imgH, int gridX, int gridY, int cLoop, enum processType t = elementWise);

template<> void
launchCudaProcess<float>(int imgW, int imgH, int gridX, int gridY, int cLoop, enum processType t)
{
	float* srcImage, *gainImage, *dstImage, *cpuSrc, *cpuGain;
	int s = imgW*imgH;

	cudaMalloc((float**)&srcImage, (s*sizeof(float)));
	cudaMalloc((float**)&gainImage, (s*sizeof(float)));
	cudaMalloc((float**)&dstImage, (s*sizeof(float)));
	cpuSrc = (float*)malloc(s*sizeof(float));
	cpuGain = (float*)malloc(s*sizeof(float));

	fillRandomNumber<float>(cpuSrc, s);
	fillRandomNumber<float>(cpuGain, s);

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

	std::vector<duration> timeDuration1;
	std::vector<duration> timeDuration2;
	for(int i = 0;i < cLoop;i++)
	{
		auto start = std::chrono::system_clock::now();
		cudaMemcpy(srcImage, cpuSrc, s*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(gainImage, cpuGain, s*sizeof(float), cudaMemcpyHostToDevice);
		auto middle = std::chrono::system_clock::now();
		launchCudaProcessFloat0(grid, block, gainImage, srcImage, dstImage, imgW);
		auto end  = std::chrono::system_clock::now();
		auto dur1 = middle - start;
		auto dur2 = end - middle;
		duration usec1 = std::chrono::duration_cast<timeResolution>(dur1).count();
		duration usec2 = std::chrono::duration_cast<timeResolution>(dur2).count();
		timeDuration1.push_back(usec1);
		timeDuration2.push_back(usec2);
	}
	std::cout << getMedian(timeDuration2) << " + " << getMedian(timeDuration1) << messageTime << s << messagePixel << '(' << imgW << 'x' << imgH << ')' << std::endl;

	cudaFree((void*)srcImage);
	cudaFree((void*)gainImage);
	cudaFree((void*)dstImage);
	free((void*)cpuSrc);
	free((void*)cpuGain);
}

template<> void
launchCudaProcess<short>(int imgW, int imgH, int gridX, int gridY, int cLoop, enum processType t)
{
	short* srcImage, *gainImage, *dstImage, *cpuSrc, *cpuGain;
	int s = imgW*imgH;

	cudaMalloc((short**)&srcImage, (s*sizeof(short)));
	cudaMalloc((short**)&gainImage, (s*sizeof(short)));
	cudaMalloc((short**)&dstImage, (s*sizeof(short)));
	cpuSrc = (short*)malloc(s*sizeof(short));
	cpuGain = (short*)malloc(s*sizeof(short));

	fillRandomNumber<short>(cpuSrc, s);
	fillRandomNumber<short>(cpuGain, s);

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


	std::vector<duration> timeDuration1;
	std::vector<duration> timeDuration2;
	switch(t)
	{
		case pack2:
			for(int i = 0;i < cLoop;i++)
			{
				auto start = std::chrono::system_clock::now();
				cudaMemcpy(srcImage, cpuSrc, s*sizeof(short), cudaMemcpyHostToDevice);
				cudaMemcpy(gainImage, cpuGain, s*sizeof(short), cudaMemcpyHostToDevice);
				auto middle = std::chrono::system_clock::now();
				launchCudaProcessHalf1(grid, block, gainImage, srcImage, dstImage, imgW);
				auto end  = std::chrono::system_clock::now();
				auto dur1 = middle - start;
				auto dur2 = end - middle;
				duration usec1 = std::chrono::duration_cast<timeResolution>(dur1).count();
				duration usec2 = std::chrono::duration_cast<timeResolution>(dur2).count();
				timeDuration1.push_back(usec1);
				timeDuration2.push_back(usec2);
			}
			break;
		default:
			for(int i = 0;i < cLoop;i++)
			{
				auto start = std::chrono::system_clock::now();
				cudaMemcpy(srcImage, cpuSrc, s*sizeof(short), cudaMemcpyHostToDevice);
				cudaMemcpy(gainImage, cpuGain, s*sizeof(short), cudaMemcpyHostToDevice);
				auto middle = std::chrono::system_clock::now();
				launchCudaProcessHalf0(grid, block, gainImage, srcImage, dstImage, imgW);
				auto end  = std::chrono::system_clock::now();
				auto dur1 = middle - start;
				auto dur2 = end - middle;
				duration usec1 = std::chrono::duration_cast<timeResolution>(dur1).count();
				duration usec2 = std::chrono::duration_cast<timeResolution>(dur2).count();
				timeDuration1.push_back(usec1);
				timeDuration2.push_back(usec2);
			}
			break;
	}
	std::cout << getMedian(timeDuration2) << " + " << getMedian(timeDuration1) << messageTime << s << messagePixel << '(' << imgW << 'x' << imgH << ')' << std::endl;

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
	launchCudaProcess<float>(1920, 1080, 16, 16, 1000);
	launchCudaProcess<short>(1920, 1080, 16, 16, 1000);
	launchCudaProcess<short>(1920, 1080, 16, 16, 1000, pack2);
	launchCudaProcess<float>(640,  480,  16, 16, 1000);
	launchCudaProcess<short>(640,  480,  16, 16, 1000);
	launchCudaProcess<short>(640,  480,  16, 16, 1000, pack2);
	launchCudaProcess<float>(3840, 2160, 16, 16, 1000);
	launchCudaProcess<short>(3840, 2160, 16, 16, 1000);
	launchCudaProcess<short>(3840, 2160, 16, 16, 1000, pack2);
	launchCudaProcess<float>(8000, 8000, 16, 16, 1000);
	launchCudaProcess<short>(8000, 8000, 16, 16, 1000);
	launchCudaProcess<short>(8000, 8000, 16, 16, 1000, pack2);

	return 0;
}
