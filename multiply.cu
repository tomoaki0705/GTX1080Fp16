#include <cuda_fp16.h>

__global__ void
cudaProcessHalf0(half *dst, half *gain, half *src, int imgW)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	int px = y*imgW+x;

	half g = gain[px];
	half i = src[px];

	dst[px] = __hmul(g, i);
}

__global__ void
cudaProcessFloat0(float *dst, float *gain, float *src, int imgW)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	int px = y*imgW+x;

	float g = gain[px];
	float i = src[px];

	dst[px] = g*i;
}

extern "C" void
launchCudaProcessHalf0(dim3 grid, dim3 block, 
					short *gain, short *imageInput, short *imageOutput, int imgW)
{
	cudaProcessHalf0<<< grid, block, 0 >>>((half*)imageOutput, (half*)gain, (half*)imageInput, imgW);
}

extern "C" void
launchCudaProcessFloat0(dim3 grid, dim3 block,
					float *gain, float *imageInput, float *imageOutput, int imgW)
{
	cudaProcessFloat0<<< grid, block, 0 >>>(imageOutput, gain, imageInput, imgW);
}

