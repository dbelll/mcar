/*
 *  cuda_row_reduction.h
 *  cuda_bandit
 *
 *  Created by Dwight Bell on 8/9/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

/*
 *	Reduce the rows of a two deimensional array, storing result in column 0
 */
__host__ void row_reduce(float *d_data, unsigned cols, unsigned rows);


/*
 *	Reduce the rows of an array using the min function, pointer to minimum row values will be
 *	placed in pd_minval and pointer to the index of the minimum value in each row will be place
 *	in pd_mincol.  The stride for both *pd_minval and *pd_mincol is the return value.
 */
__host__ unsigned row_argmin(float *d_data, unsigned cols, unsigned rows, float **pd_minval, unsigned **pd_mincol);

#define BIG_FLOAT 9.99e49 