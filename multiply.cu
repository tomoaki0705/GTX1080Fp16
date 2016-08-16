#include <cuda_fp16.h>

extern "C" void
launchCudaProcessHalf(dim3 grid, dim3 block, int sbytes,
					short *gain, float *imageInput, float *imageOutput);

extern "C" void
launchCudaProcessFloat(dim3 grid, dim3 block, int sbytes,
					float *gain, float *imageInput, float *imageOutput);

__global__ void
cudaProcessHalf(half2 *dst, half2 *gain, half2 *src, int imgW)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	int px = (y*imgW+x);

	half2 g = gain[px];
	half2 i = src[px];

	dst[px] = __hmul2(g, i);
}
