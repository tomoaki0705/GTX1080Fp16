#include <iostream>

extern "C" void
launchCudaProcessHalf(dim3 grid, dim3 block, int sbytes,
					short *gain, uchar *imageInput, uchar *imageOutput);

extern "C" void
launchCudaProcessFloat(dim3 grid, dim3 block, int sbytes,
					float *gain, uchar *imageInput, uchar *imageOutput);

int main()
{
	return 0;
}
