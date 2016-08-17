#include <iostream>
#include <vector_types.h>
#include <chrono>
#include "types.h"

extern "C" void
launchCudaProcessHalf(dim3 grid, dim3 block, int sbytes,
					short *gain, short *imageInput, short *imageOutput);

extern "C" void
launchCudaProcessFloat(dim3 grid, dim3 block, int sbytes,
					float *gain, float *imageInput, float *imageOutput);

int main()
{
	float* imageFloat = new float[1920*1080];
	float* gainFloat = new float[1920*1080];
	float* dstFloat = new float[1920*1080];
	short* imageHalf = new short[1920*1080];
	short* gainHalf = new short[1920*1080];
	short* dstHalf = new short[1920*1080];

	auto start = std::chrono::system_clock::now();
	auto end  = std::chrono::system_clock::now();
	auto dur = end - start;
	auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();

	std::cout << msec << "[ms] passed" << std::endl;

	delete [] imageFloat;
	delete [] gainFloat;
	delete [] dstFloat;
	delete [] imageHalf;
	delete [] gainHalf;
	delete [] dstHalf;
	return 0;
}
