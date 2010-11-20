/*
 *  cuda_utils.h
 *
 *  Created by Dwight Bell on 5/14/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */

#ifndef __CUDA_UTILS_H__
#define __CUDA_UTILS_H__

#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>


// Macros for calculating timing values.
// Caller must supply a pointer to unsigned int when creating a timer,
// and the unsigned int for other timer calls.
void CREATE_TIMER(unsigned int *p_timer);
void START_TIMER(unsigned int timer);
float STOP_TIMER(unsigned int timer, char *message);
void DELETE_TIMER(unsigned int timer);

// random number generation
unsigned TausStep(unsigned &z, int S1, int S2, int S3, unsigned M);
unsigned LCGStep(unsigned &z, unsigned A, unsigned C);
float HybridTaus();

#endif