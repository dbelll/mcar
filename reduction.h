/*
 *  cuda_row_reduction.h
 *  cuda_bandit
 *
 *  Created by Dwight Bell on 8/9/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

/*
 *	Reduce the rows of a two dimensional array, storing result in column 0
 *
 *	Warning, destructive to original data.
 */
__host__ void row_reduce(float *d_data, unsigned cols, unsigned rows);


/*
 *	Reduce the columns of a two dimensional array, storing result in the provided device vector
 *  If the output array is NULL, one will be allocated.
 *	Returns the pointer to the total array.
 *	col_reduce_x_k() multiplies the resulting totals by a factor.
 */

__host__ float *col_reduce(float *d_data, float *d_totals, unsigned cols, unsigned rows);
__host__ float *col_reduce_x_k(float *d_data, float *d_totals, unsigned cols, unsigned rows, float k);
__host__ float *col_avg(float *d_data, float *d_totals, unsigned cols, unsigned rows);

__host__ float clean_reduce(float *d_data, unsigned n);

// add two vectors x and y storing result in x.  x has stride of 1, and y's stride is specified
__host__ void vsum(float *x, float *y, unsigned n, unsigned stride_y);

/*
 *	Reduce the rows of an array using the min function, pointer to minimum row values will be
 *	placed in pd_minval and pointer to the index of the minimum value in each row will be place
 *	in pd_mincol.  The stride for both *pd_minval and *pd_mincol is the return value.
 */
__host__ unsigned row_argmin(float *d_data, unsigned cols, unsigned rows, float **pd_minval, unsigned **pd_mincol);
__host__ unsigned row_argmin2(float *d_data, unsigned cols, unsigned rows, float *d_minval, unsigned *d_mincol);
__host__ unsigned row_argmax(float *d_data, unsigned cols, unsigned rows, float **pd_minval, unsigned **pd_mincol);

#define BIG_FLOAT 9.99e49
#define SMALL_FLOAT -9.99e49

