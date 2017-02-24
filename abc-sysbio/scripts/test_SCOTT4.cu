#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#define NCOLS 163317 // number of columns
#define NROWS 8 // number of rows
#define nTPB 1024  // Threads per Block. nTPB should be a power-of-2
#define MAX_BLOCKS_X ((NCOLS/nTPB)+1) // # of blocks I will launch

#define FLOAT_MIN -1.0f // lowest anticipated number of the data. Values in array will be compared with this and updated with the larger one


__device__ volatile float blk_vals[NROWS][MAX_BLOCKS_X];
__device__ volatile int   blk_idxs[NROWS][MAX_BLOCKS_X];
// blk_vals and blk_idxs are the results obtained from reduction within each block.
// after 1st reduction, each row will have blk_vals[MAX_BLOCKS_X] array and blk_idxs[MAX_BLOCKS_X]
// these will be passed to the 2nd kernel

__global__ void max_idx_kernel_reduction_within_block(const float *data, const int xSize, const int ySize){  // first kernel. Reduction within blocks
  __shared__ volatile float vals[nTPB]; // Total amount of shared memory per block: 49152 bytes (50 KB). 1024 gives ~ 4KB for single.
  __shared__ volatile int idxs[nTPB]; // ~ 4 KB for single, when nTPB is 1024. each block will have both indices and values

  int idx = threadIdx.x+blockDim.x * blockIdx.x; // idx in the x direction
  int idy = blockIdx.y;
  float my_val = FLOAT_MIN; // lowest possible number
  int my_idx = -1;  // to check whether you DID perform the kernel. Again, it's the idx in the x dir.

  // sweep from global memory
  while (idx < xSize){   // this ensures you don't go out the size of the array's x direction
    float temp = data[idy*xSize+idx];
    if (temp > my_val) {my_val = temp; my_idx = idx;}
    // compare with my_val, and put the bigger value into my_val for next comparison. my_idx is 0 index based
    idx += blockDim.x*gridDim.x;}
                                                                 // until here takes about 6 ms !! very fast!!
  // populate shared memory: takes ~ 270 ms
  vals[threadIdx.x] = my_val;  // put the computed max value for each thread into the shared memory. -> this is the bottleneck!!
  idxs[threadIdx.x] = my_idx;  // do this for index as well -> this is also slow!!

  __syncthreads();

  // sweep in shared memory
  for (int i = (nTPB>>1); i > 0; i>>=1){
    if (threadIdx.x < i)    // the first half threads of the block
      if (vals[threadIdx.x] < vals[threadIdx.x + i]) {vals[threadIdx.x] = vals[threadIdx.x+i]; idxs[threadIdx.x] = idxs[threadIdx.x+i]; }
                            // the above is comparing shared memory of threadIdx.x with shared memory of threadIdx.x + i.
                            // then puts the larger value into shared memory of threadIdx.x
    __syncthreads();}       // so now in each block, shared memory's first element (index 0) is the max value and max value index


  // perform block-level reduction
  if (!threadIdx.x){    // at the shared memory, only the first element (index 0) (actually 2 elements in the first index. max value, and max value index) is what we need.
      blk_vals[blockIdx.y][blockIdx.x] = vals[0]; // For each window (single x row), the first elements of the blocks are stored into the blk_vals[windowNumber][:]
                                                // remember, this is a global variable.
      blk_idxs[blockIdx.y][blockIdx.x] = idxs[0]; // and the max value index

  __syncthreads();
}

}

  // originally the following kernel was in the 1st kernel, performed by the last block. So just use one block for this.
__global__ void max_idx_kernel_final(int *result_maxInd, float *result_maxVal){

  __shared__ volatile float vals[nTPB]; //  Total amount of shared memory per block: 49152 bytes (50 KB). 1024 gives ~ 4KB for single.
  __shared__ volatile int idxs[nTPB]; // ~ 4 KB for single, when nTPB is 1024. each block will have these variables!! (vals and idxs)

  int idx = threadIdx.x;
  int idy = blockIdx.y;
  float my_val = FLOAT_MIN;
  int my_idx = -1;  // remember, these are local variables, so each thread has this variable. This local variable is independent from other thread's local variable
  while (idx < MAX_BLOCKS_X ){                                                          // ?? confused whether it should be gridDim.x (actual # of blocks launched) or MAX_BLOCKS_X (# of elements in x dir of the global array blk_vals)
    float temp = blk_vals[idy][idx];
    if (temp > my_val)
        {my_val = temp; my_idx = blk_idxs[idy][idx]; }
    idx += blockDim.x;} // all threads in this single block (single in the x dir) are working, so you should loop over blockDim.x.
                      // Imagine where gridDim.x (# of blocks) is huge so that you need to loop over to get the max value and index
                      // After this, each thread in the block has a local variable (max value and max value index).
                      // So far it was sort of a reduction, but instead of pairing values we just looped over the blk_vals and blk_idxs
  // populate shared memory
  idx = threadIdx.x;
  vals[idx] = my_val;   // This is now shared memory. This is because reduction requires comparison between different elements
  idxs[idx] = my_idx;   // my_idx value is 0 based. This is done for all blocks (in the y direction)
  __syncthreads();
  // Now the final task is to do reduction for all threads in our single block (single block in the x dir, NROWS blocks in the y dir)!

// sweep in shared memory
  for (int i = (nTPB>>1); i > 0; i>>=1) {
    if (idx < i) // the first half threads of the block
      if (vals[idx] < vals[idx + i]) {vals[idx] = vals[idx+i]; idxs[idx] = idxs[idx+i]; }
    __syncthreads();} // now all the results are in threadIdx.x == 0 for each block (there are NROWS blocks in the y dir)
  // 0th thread. the results are in shared memory, not the local memory, so any thread could do the following. We just selected the 0th thread for no reason. If several threads try to do this, that would be a problem, since we'll have to wait for them

  if(!threadIdx.x){
        result_maxInd[idy] = idxs[0]; // the final result for each row goes into the corresponding position (blockIdx.y)
        result_maxVal[idy] = vals[0];
      }
}


int main(){

    dim3 grids(MAX_BLOCKS_X, NROWS); //(160,8,1)
    dim3 threads(nTPB,1); //(1024,1,1)
    dim3 grids2(1,NROWS); //(1,8,1)
    dim3 threads2(nTPB); //(1024,1,1)

    float *d_vector, *h_vector;

    h_vector = (float*)malloc(NROWS * NCOLS * sizeof(float));
    memset(h_vector, 0, NROWS*NCOLS*sizeof(float));

    for (int i =  0; i < NROWS; i++){
      h_vector[i*NCOLS + i] = 10.0f;  // create definite max element per row
      printf("%f\n", h_vector[i*NCOLS + i]);
    }
    cudaMalloc(&d_vector, NROWS * NCOLS * sizeof(float));
    cudaMemcpy(d_vector, h_vector, NROWS * NCOLS * sizeof(float), cudaMemcpyHostToDevice);

    //d_vector is a pointer on the device pointing to the beginning of the vector, containing nrElements floats.

    int *max_index;
    float *max_val;
    int *d_max_index;
    float *d_max_val;

    max_index = (int*)malloc(NROWS * sizeof(int));
    max_val = (float*)malloc(NROWS * sizeof(float));
    cudaMalloc((void**)&d_max_index, NROWS * sizeof(int));
    cudaMalloc((void**)&d_max_val, NROWS * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    max_idx_kernel_reduction_within_block<<<grids, threads>>>(d_vector, NCOLS, NROWS);
    max_idx_kernel_final<<<grids2,threads2>>>(d_max_index, d_max_val);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float et;
    cudaEventElapsedTime(&et, start, stop);
    printf("elapsed time: %fms\n", et);

    cudaMemcpy(max_index, d_max_index, NROWS * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_val, d_max_val, NROWS * sizeof(float), cudaMemcpyDeviceToHost);

    for(int z=0;z<NROWS;z++)
      printf("%d  ",max_index[z]);

    printf("\n\n\n");

    for(int z=0;z<NROWS;z++)
      printf("%f  ",max_val[z]);
    printf("\n");
    return 0;
}
