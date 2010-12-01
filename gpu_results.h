/*
 *  gpu_results.h
 *  mcar
 *
 *  Created by Dwight Bell on 11/30/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */
#include "mcar.h"


/*
	A growable list of result information from GPU learning
 */

typedef struct {
	unsigned allocated;
	unsigned next;
	unsigned dumpUpdates;		// flag to indicate print out each update as it happens
//	unsigned num_wgts;
//	unsigned num_agents;		// needed for stride of theta values in AGENT_DATA
	GPU_RESULTS *results;
//	float *theta;
} GPU_RESULT_LIST;


void prepare_GPU_result_list(unsigned size_est, unsigned dumpUpdates);
void delete_GPU_result_list();

void add_to_GPU_result_list(AGENT_DATA *agGPU, unsigned iBest, unsigned t);
void dump_GPU_result_list();
