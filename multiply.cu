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

	i = __hmul(g, i); // 1
	i = __hmul(g, i); // 2
	i = __hmul(g, i); // 3
	i = __hmul(g, i); // 4
	i = __hmul(g, i); // 5
	i = __hmul(g, i); // 6
	i = __hmul(g, i); // 7
	i = __hmul(g, i); // 8
	i = __hmul(g, i); // 9
	dst[px] = __hmul(g, i); // 10
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

	i = g*i; // 1
	i = g*i; // 2
	i = g*i; // 3
	i = g*i; // 4
	i = g*i; // 5
	i = g*i; // 6
	i = g*i; // 7
	i = g*i; // 8
	i = g*i; // 9
	dst[px] = g*i; // 10
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

	i = __hmul2(g, i); // 1
	i = __hmul2(g, i); // 2
	i = __hmul2(g, i); // 3
	i = __hmul2(g, i); // 4
	i = __hmul2(g, i); // 5
	i = __hmul2(g, i); // 6
	i = __hmul2(g, i); // 7
	i = __hmul2(g, i); // 8
	i = __hmul2(g, i); // 9
	dst[px] = __hmul2(g, i); // 10
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
