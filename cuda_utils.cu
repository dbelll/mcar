/*
 *  cuda_utils.cu
 *
 *  Created by Dwight Bell on 5/14/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

#include "cuda_utils.h"

void CREATE_TIMER(unsigned int *p_timer){ cutilCheckError(cutCreateTimer(p_timer)); }
void START_TIMER(unsigned int timer){ 
	cutilCheckError(cutResetTimer(timer));
	cutilCheckError(cutStartTimer(timer)); 
}
float STOP_TIMER(unsigned int timer, char *message){
	cutilCheckError(cutStopTimer(timer));
	float elapsed = cutGetTimerValue(timer);
	if (message) printf("%12.3f ms for %s\n", elapsed, message);
	return elapsed;
}
void DELETE_TIMER(unsigned int timer){ cutilCheckError(cutDeleteTimer(timer)); }


/*
 *	Random nubmers on the GPU
 */

__device__ unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M)
{
	unsigned b = (((z << S1) ^ z) >> S2);
	return z = (((z & M) << S3) ^ b);
}

__device__ unsigned LCGStep(unsigned &z, unsigned A, unsigned C)
{
	return z = (A*z + C);
}

/* generate a random number, uses an array of 4 unsigned ints */
__device__ float HybridTaus(unsigned *z)
{
	return 2.3283064365387e-10 * (
		TausStep(z[0], 13, 19, 12, 4294967294UL) ^
		TausStep(z[1], 2, 25, 4, 4294967288UL) ^
		TausStep(z[2], 3, 11, 17, 4294967280UL) ^
		LCGStep(z[3], 16654525, 1013904223UL)
	);
}

