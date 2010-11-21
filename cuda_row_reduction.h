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
