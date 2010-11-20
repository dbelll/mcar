//
//  main.c
//  mcar
//
//  Created by Dwight Bell on 11/20/10.
//  Copyright dbelll 2010. All rights reserved.
//


#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "mcar.h"

int main(int argc, char **argv)
{
	int n = 1000000;
	float *x = (float *)malloc(n * sizeof(float));

	srand(100);
	for (int i = 0; i < n; i++) {
		x[i] = (float)rand() / (float)RAND_MAX;
	}
	
	float *x2 = (float *)malloc(n * sizeof(float));
	memcpy(x2, x, n * sizeof(float));
	
	printf("before...\n");
	for (int i = 0; i < 10; i++) {
		printf("x[%d] = %f\n", i, x[i]);
	}

	gpu_operation(n, x);
	
	printf("after...\n");
	for (int i = 0; i < 10; i++) {
		printf("x[%d] = %f\n", i, x[i]);
	}
	
	cpu_operation(n, x2);
	
	float err = 0.0f;
	float e;
	for (int i = 0; i < n; i++) {
		e = x[i] - x2[i];
		if (e < 0) e = -e;
		if (e > err) err = e;
	}
	printf("max error is %f\n", err);
	
	return 0;
}
