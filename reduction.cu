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
 
// remove this definition when BLOCK_SIZE is defined elsewhere
// and add the proper #include reference
//#define BLOCK_SIZE 256
#include "mcar.h"

//#define __DEBUG_REDUCTION__

#include <cuda.h>
#include "cutil.h"

#include "cuda_utils.h"
#include "reduction.h"
#include <stdio.h>
#include <math.h>

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
  // starts at 1.  On the second kernel invocation, the values to be summed are 
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


__global__ void argmin_kernel(float *g_data, int in_cols, int g_stride, float *d_minval, unsigned *d_mincol, int out_cols)
{
	// adjust g_data to point to the correct row for this block
	// (row number is the y index of the block)
	g_data += in_cols * blockIdx.y;
	
	// adjust d_minval and d_mincol to point to the correct row for this block
	d_minval += out_cols * blockIdx.y;
	d_mincol += out_cols * blockIdx.y;
	
	// Define shared memory
	__shared__ float s_data[BLOCK_SIZE];
	__shared__ unsigned s_index[BLOCK_SIZE];


	// Load the shared memory (the first reduction occurs while loading shared memory)

	// index into shared memory for this thread
	int s_i = threadIdx.x;

	// index into global memory for this thread
	int g_i = g_stride * (s_i + (blockIdx.x * blockDim.x) * 2);

	int half = blockDim.x;   // half equals 1/2 the number of values left to be reduced

	// if g_i points to real data, copy it to shared memory, otherwise plug in a 0.0 value
	if(g_i < in_cols){
		s_data[s_i] = g_data[g_i];
		s_index[s_i] = g_i;
	}else{
		s_data[s_i] = BIG_FLOAT;
	}

	// if the value a BLOCK_SIZE away is real data add it to shared
	//  if((g_i + half*g_stride) < cols*g_stride)
	if((g_i + half*g_stride) < in_cols){
		if (s_data[s_i] > g_data[g_i + half * g_stride]){
			s_data[s_i] = g_data[g_i + half * g_stride];
			s_index[s_i] = g_i + half * g_stride;
		}
	}
	
	half /= 2;
	__syncthreads();   // make sure all threads are done with the first reduction

	// Do sum reduction from shared memory
	while(half > 0){
		if(s_i < half){
			if (s_data[s_i] > s_data[s_i + half]){
				s_data[s_i] = s_data[s_i + half];
				s_index[s_i] = s_index[s_i + half];
			}
		}
		half /= 2;
		__syncthreads();
	}

	// Store just the result back to the output memory
	if(threadIdx.x == 0){
		d_minval[blockIdx.x] = s_data[0];
		d_mincol[blockIdx.x] = s_index[0];
	}

  return;
}


__global__ void argmin_kernel2(float *g_data, unsigned *g_index, int cols, int g_stride, int orig_cols)
{
	// adjust g_data and g_index to point to the correct row for this block
	// (row number is the y index of the block)
	g_data += orig_cols * blockIdx.y;
	g_index += orig_cols * blockIdx.y;
		
	// Define shared memory
	__shared__ float s_data[BLOCK_SIZE];
	__shared__ unsigned s_index[BLOCK_SIZE];


	// Load the shared memory (the first reduction occurs while loading shared memory)

	// index into shared memory for this thread
	int s_i = threadIdx.x;

	// index into global memory for this thread
	int g_i = g_stride * (s_i + (blockIdx.x * blockDim.x) * 2);

	int half = blockDim.x;   // half equals 1/2 the number of values left to be reduced

	// if g_i points to real data, copy it to shared memory, otherwise plug in a 0.0 value
	if(g_i < orig_cols){
		s_data[s_i] = g_data[g_i];
		s_index[s_i] = g_index[g_i];
	}else{
		s_data[s_i] = BIG_FLOAT;
	}

	// if the value a BLOCK_SIZE away is real data add it to shared
	if((g_i + half*g_stride) < orig_cols){
		if (s_data[s_i] > g_data[g_i + half * g_stride]){
			s_data[s_i] = g_data[g_i + half * g_stride];
			s_index[s_i] = g_index[g_i + half * g_stride];
		}
	}

	half /= 2;
	__syncthreads();   // make sure all threads are done with the first reduction

	// Do argmin reduction from shared memory
	while(half > 0){
		if(s_i < half){
			if (s_data[s_i] > s_data[s_i + half]){
				s_data[s_i] = s_data[s_i + half];
				s_index[s_i] = s_index[s_i + half];
			}
		}
		half /= 2;
		__syncthreads();
	}

	// Store just the result back to g_data and g_index
	if(threadIdx.x == 0){
		g_data[g_i] = s_data[0];
		g_index[g_i] = s_index[0];
	}

  return;
}




// determine the minimum value by row for an array of values, recording which column had the minimum
// the d_min and d_mincol pointers will be filled in pointing to the results on the device
// The stride of d_min and d_mincol is the return value.  It will be equal to 1 + (cols-1)/(2*BLOCK_SIZE)
// *d_min and *d_mincol must be freed by caller.
__host__ unsigned row_argmin(float *d_data, unsigned cols, unsigned rows, float **pd_minval, unsigned **pd_mincol)
{
	int stride = 1;
	int orig_cols = cols;
	dim3 blockDim(BLOCK_SIZE);
	dim3 gridDim;
	
	// allocate the output arrays
	unsigned col_after_one = 1 + (cols-1)/(2*BLOCK_SIZE);
	*pd_minval = (float *)device_allocf(col_after_one * rows);
	*pd_mincol = (unsigned *)device_allocui(col_after_one * rows);
	float *d_minval = *pd_minval;
	unsigned *d_mincol = *pd_mincol;
	
	unsigned firstTimeThrough = 1;
	while (cols > 1) {
		// each block will handle the reduction of 2*BLOCK_SIZE columns
		// values to be reduced are separated by stride memory locations
		gridDim.x = 1 + (cols-1)/(2*BLOCK_SIZE);
		if (gridDim.x > 65535) printf("[ERROR] Too many columns for row_argmin\n");
		gridDim.y = rows;
		
		if (firstTimeThrough) {
			// reduce the values in d_data and store the results in *d_min and put the indexes in *d_mincol
			unsigned outCols = 1 + (cols-1) / (2*BLOCK_SIZE);
#ifdef __DEBUG_REDUCTION__
			printf("argmin_kernel, gridDim=(%dx%d), blockDim=(%dx%d), cols=%d, stride=%d, outCols = %d\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, cols, stride, outCols);
#endif
			argmin_kernel<<<gridDim, blockDim>>>(d_data, cols, stride, d_minval, d_mincol, outCols);
#ifdef __DEBUG_REDUCTION__
			device_dumpf("minvals after argmin_kernel", d_minval, rows, col_after_one);
			device_dumpui("mincols after argmin_kernel", d_mincol, rows, col_after_one);
#endif
		}else {
			// reduce the values in *d_min and indexes in *d_mincol destructively
#ifdef __DEBUG_REDUCTION__
			printf("argmin_kernel2, gridDim=(%dx%d), blockDim=(%dx%d), cols=%d, stride=%d, orig_cols = %d\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, cols, stride, orig_cols);
#endif
			argmin_kernel2<<<gridDim, blockDim>>>(d_minval, d_mincol, cols, stride, orig_cols);
#ifdef __DEBUG_REDUCTION__
			device_dumpf("minvals after argmin_kernel2", d_minval, rows, orig_cols);
			device_dumpui("mincols after argmin_kernel2", d_mincol, rows, orig_cols);
#endif
		}
		
		CUDA_SAFE_CALL(cudaThreadSynchronize());
		CUT_CHECK_ERROR("arg_min_kernel execution failed");
		
		cols = 1 + (cols-1) / (2*BLOCK_SIZE);
		
		if (firstTimeThrough) {
			firstTimeThrough = 0;
			orig_cols = cols;
		}else {
			stride *= 2*BLOCK_SIZE;
		}
	}
	return col_after_one;
}

#endif