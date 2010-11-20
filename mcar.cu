//
//  mcar.cu
//  mcar
//
//  Created by Dwight Bell on 11/20/10.
//  Copyright dbelll 2010. All rights reserved.
//

#include <cuda.h>
#include "cutil.h"

#include "mcar.h"
#include "cuda_utils.h"

#pragma mark CUDA Kernels
__global__ void kernel_operation(int n, float *x)
{
	int i = threadIdx.x + (blockIdx.x * blockDim.x) + (blockIdx.y * gridDim.x * blockDim.x);
	if (i >= n) return;
	
	x[i] = sqrt(x[i]);
}


#pragma mark CPU front end
__host__ void cpu_operation(int n, float *x)
{
	unsigned int timer;
	CREATE_TIMER(&timer);
	START_TIMER(timer);
	for (int i = 0; i < n; i++) {
		x[i] = sqrt(x[i]);
	}
	STOP_TIMER(timer, "cpu operation");
}

#pragma mark GPU front end
void gpu_operation(int n, float *x)
{
	unsigned int timer;
	CREATE_TIMER(&timer);

	// copy data to device
	START_TIMER(timer);
	float *d_x = NULL;
	int size = n * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_x, size));
	CUDA_SAFE_CALL(cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice));
	cudaThreadSynchronize();
	STOP_TIMER(timer, "copy data to device");


	// calculate size of block and grid
	int tot_threads = 32 * (1 + (n-1)/32);	// smallest multiple of 32 >= n
	dim3 blockDim(min(512,tot_threads));
	dim3 gridDim(1 + (tot_threads -1)/512, 1);
	if (gridDim.x > 65535) {
		gridDim.y = 1 + (gridDim.x-1) / 65535;
		gridDim.x = 1 + (gridDim.x-1) / gridDim.y;
	}
	
	// Do the operation on the device
	START_TIMER(timer);	
	kernel_operation<<<gridDim, blockDim>>>(n, d_x);
	cudaThreadSynchronize();
	STOP_TIMER(timer, "GPU operations");

	// Check if kernel execution generated an error
	CUT_CHECK_ERROR("Kernel execution failed");
	
	
	// copy results back to host
	START_TIMER(timer);
	CUDA_SAFE_CALL(cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost));
	cudaThreadSynchronize();
	STOP_TIMER(timer, "copy data back to host");
	
	
}