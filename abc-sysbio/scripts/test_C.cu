#include <stdio.h>

__global__ void reduce0(int *g_idata, int *g_odata){
		__shared__ int sdata[128];
		unsigned int tid = threadIdx.x;
		unsigned int i = blockIdx.x*(blockDim.x*2)+threadIdx.x;

		sdata[tid] = g_idata[i]+g_idata[i+blockDim.x];
		__syncthreads();


		for(unsigned int s=blockDim.x/2; s>32; s>>=1){
			if (tid < s){
				sdata[tid] += sdata[tid+s];
			}
			__syncthreads();
		}

		if (tid < 32){
			sdata[tid] += sdata[tid+32];
			sdata[tid] += sdata[tid+16];
			sdata[tid] += sdata[tid+8];
			sdata[tid] += sdata[tid+4];
			sdata[tid] += sdata[tid+2];
			sdata[tid] += sdata[tid+1];
		}

		if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main(void)
{
	int N = 512;
	int *x, *y, *d_x, *d_y;

	dim3 block(128,1,1);
	dim3 grid(2,1,1);

	x = (int*)malloc(N*sizeof(int));
	y = (int*)malloc(2*sizeof(int));

	cudaMalloc(&d_x, N*sizeof(int)); 
	cudaMalloc(&d_y, 2*sizeof(int));

	for (int i = 1; i < (N+1); i++) {
		x[i-1] = i;
	}

	y[0] = 0;
	y[1] = 0;

	cudaMemcpy(d_x, x, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, 2*sizeof(int), cudaMemcpyHostToDevice);

	reduce0<<<grid, block>>>(d_x, d_y);

	cudaMemcpy(y, d_y, 2*sizeof(int), cudaMemcpyDeviceToHost);

	printf("%d\n", y[0]+y[1]);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
}