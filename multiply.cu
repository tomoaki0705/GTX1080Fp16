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

__global__ void
cudaProcessHalf1(half2 *dst, half2 *gain, half2 *src, int imgW)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;
	int px = (y*imgW/2)+x;

	half2 g = gain[px];
	half2 i = src[px];

	dst[px] = __hmul2(g, i);
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

extern "C" void
launchCudaProcessHalf1(dim3 grid, dim3 block, 
					short *gain, short *imageInput, short *imageOutput, int imgW)
{
	cudaProcessHalf1<<< grid, block, 0 >>>((half2*)imageOutput, (half2*)gain, (half2*)imageInput, imgW);
}
