#ifndef __CUDA_REDUCTION_CU__
#define __CUDA_REDUCTION_CU__

/*
 * vector reduction kernel for reducing rows of a 2D array
 *
 * cols values to be reduced in each row, spaced g_stride apart in global memory
 *
 * This kernel will reduce a range of vector values of size 2 x BLOCK_SIZE.
 * The values are stored in global memory with a spacing specified by g_stride.
 * First, the values are copied to shared memory to an array of size 2 x BLOCK_SIZE.
 * Then this kernel calculates the sum of the 2 x BLOCK_SIZE values and stores
 * the sum back in global memory at g_data.
 *
 *	The maximum number of colums is 512*2*65535 = 67107840
 *	The maximum number of rows is 65535
 *	The grid's y dimension is used for the rows
 */
 
#include <cuda.h>
#include "cutil.h"

#include "cuda_utils.h"
#include "pole.h"
#include "cuda_row_reduction.h"
#include <stdio.h>

__global__ void row_reduction(float *g_data, int cols, int g_stride, int orig_cols)
{

	// adjust g_data to point to the correct row
	g_data += orig_cols * blockIdx.y;
	
  // Define shared memory
  __shared__ float s_data[BLOCK_SIZE];


  // Load the shared memory (the first reduction occurs while loading shared memory)

  // index into shared memory for this thread
  int s_i = threadIdx.x;

  // index into global memory for this thread
  int g_i = g_stride * (s_i + (blockIdx.x * blockDim.x) * 2);

  int half = BLOCK_SIZE;   // half equals 1/2 the number of values left to be reduced

  // if g_i points to real data copy it to shared memory, otherwise plug in a 0.0 value
//  if(g_i < cols*g_stride)
  if(g_i < orig_cols)
    s_data[s_i] = g_data[g_i];
  else
    s_data[s_i] = 0.0;

  // if the value a BLOCK_SIZE away is real data add it to shared
//  if((g_i + half*g_stride) < cols*g_stride)
  if((g_i + half*g_stride) < orig_cols)
    s_data[s_i] += g_data[g_i + half*g_stride];

  half /= 2;
  __syncthreads();   // make sure all threads are done with the first reduction

  // Do sum reduction from shared memory
  while(half > 0){
    if(s_i < half)
      s_data[s_i] += s_data[s_i+half];
    half /= 2;
    __syncthreads();
  }

  // Store just the total back to global memory
  if(threadIdx.x == 0)
    g_data[g_i] = s_data[s_i];

  return;
}


// reduce an aribitrary size 2D array on the device by rows, leaving result in column 0
__host__ void row_reduce(float *d_data, unsigned cols, unsigned rows)
{
  int stride = 1;
	int orig_cols = cols;

  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim;  // calculated inside while loop below

  while(cols > 1){
  
  //
  // Each invocation of the kernel will reduce ranges of 2 x BLOCK_SIZE values.
  // If the vector size is more than 2 x BLOCK_SIZE, then the kernel must be called
  // again.  Initially, the values to be reduced are next to each other, so g_stride
  // starts at 1.  On the second kernel invoaction, the values to be summed are 
  // (2*BLOCK_SIZE) elements apart.  On the 3rd invocation they are (2*BLOCK_SIZE)^2
  // apart, etc.  g_stride is multipled by (2*BLOCK_SIZE) after each kernel invocation.
  // 

    // First, assume a one-dimensional grid of blocks
    gridDim.x = 1 + (cols-1)/(2*BLOCK_SIZE);
	// y-dimension of grid is used for rows
    gridDim.y = rows;

    // if more than 65535 blocks then there is a problem
    if(gridDim.x > 65535){
		printf("[ERROR] Too many columns!! for row_reduce\n");
    }

    // print information for each invocation of the kernel
//	printf("[row_reduce]\n");
//    printf("threads per block is %d x %d x %d\n", blockDim.x, blockDim.y, blockDim.z);
//    printf("blocks per grid is %d x %d\n", gridDim.x, gridDim.y);
//    printf("reduction with num_elements = %d, g_stride = %d\n", cols, stride);

    // invoke the kernel
    row_reduction<<<gridDim, blockDim>>>(d_data, cols, stride, orig_cols);

    // wait for all blocks to finish
    CUDA_SAFE_CALL(cudaThreadSynchronize());

	// Check if kernel execution generated an error
	CUT_CHECK_ERROR("Kernel execution failed");

    // calculate the number of values remaining.
    cols = 1 + (cols-1)/(2*BLOCK_SIZE);

    // adjust the distance between sub-total values
    stride *= (2*BLOCK_SIZE);
  }
}

#endif