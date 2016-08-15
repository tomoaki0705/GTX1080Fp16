

extern "C" void
launchCudaProcessHalf(dim3 grid, dim3 block, int sbytes,
					short *gain, float *imageInput, float *imageOutput);

extern "C" void
launchCudaProcessFloat(dim3 grid, dim3 block, int sbytes,
					float *gain, float *imageInput, float *imageOutput);

// __half2 __hmul2(const __half2 a, const __half2 b);

__global__ void
cudaProcessHalf(float *dst, short *gain, float *src, int imgW)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bw = blockDim.x;
	int bh = blockDim.y;
	int x = blockIdx.x*bw + tx;
	int y = blockIdx.y*bh + ty;

	half a = gain[y*imgw+x];
	float gain;
	gain = __half2float(a);

	float b = (float)src[y*imgw+x];

	dst[y*imgw+x] = b * gain;
}
