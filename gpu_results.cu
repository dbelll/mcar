/*
 *  gpu_results.c
 *  mcar
 *
 *  Created by Dwight Bell on 11/30/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */
#include <stdlib.h>
#include <stdio.h>

#include "cutil.h"
#include "gpu_results.h"


static GPU_RESULT_LIST *_rl = NULL;

/*
	Allocate GPU_RESULT_LIST and initial memory
*/
void prepare_GPU_result_list(unsigned size_est, unsigned dumpUpdates)
{
	if (_rl) delete_GPU_result_list();
	_rl = (GPU_RESULT_LIST *)malloc(sizeof(GPU_RESULT_LIST));
	_rl->allocated = size_est;
	_rl->dumpUpdates = dumpUpdates;
	_rl->next = 0;
	_rl->results = (GPU_RESULTS *)malloc(_rl->allocated * sizeof(GPU_RESULTS));
//	_rL->theta = (float *)malloc(_rl->allocated * _rl->num_wgts * sizeof(float));
}

/*
	double the capicity of the gpu result list
*/
void grow_gpu_result_list()
{
	// double the size of the result data
	_rl->allocated *= 2;
	_rl->results = (GPU_RESULTS *)realloc(_rl->results, _rl->allocated * sizeof(GPU_RESULTS));
//	_rl->thetas = (GPU_RESULTS *)realloc(_rl->results, _rl->allocated * _rl->num_wgts * sizeof(float));
}

// add this agent to the new results array
void add_to_GPU_result_list(AGENT_DATA *agGPU, unsigned iBest, unsigned t, float fitness)
{
	if (!_rl) exit(-1);
	
	// increase the size of the list, if necessary
	if (_rl->next == _rl->allocated) grow_gpu_result_list();
	
	// get a pointer to the next available GPU_RESULT structure
	GPU_RESULTS *gpur = _rl->results + _rl->next;
	
	// fill in the structure
	gpur->time_step = t;
	gpur->agent = iBest;
//	CUDA_SAFE_CALL(cudaMemcpy(&gpur->fitness, agGPU->fitness + iBest, sizeof(float), cudaMemcpyDeviceToHost));
	gpur->fitness = fitness;

	if (_rl->dumpUpdates) 
		printf("\n--> %4d is new best agent with fitness of%8.2f   ", iBest, gpur->fitness / NUM_TOT_DIV);

	_rl->next += 1;
}

void delete_GPU_result_list()
{
	if (_rl) {
		if (_rl->results){ free(_rl->results); _rl->results = NULL;}
		free(_rl); _rl = NULL;
	}
}

void dump_GPU_result_list()
{
	printf("GPU_result_list:\n");
	for (int i = 0; i < _rl->next; i++) {
		GPU_RESULTS *gpur = _rl->results + i;
		printf("[timestep =%9d][agent =%4d][fitness =%8.3f]\n", gpur->time_step, gpur->agent, gpur->fitness / NUM_TOT_DIV);
	}
}

float last_fitness_on_GPU_result_list()
{
	if (_rl->next == 0) return 9999.0f;
	return _rl->results[_rl->next-1].fitness / NUM_TOT_DIV;
}

unsigned last_agent_on_GPU_result_list()
{
	if (_rl->next == 0) return 999999;
	return _rl->results[_rl->next-1].agent;
}

