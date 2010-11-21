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
#include "cuda_rand.cu"

// parameters stored in global structure for CPU
static PARAMS _p;

// Initial global seeds used to ensure identical random variables each run on all machines
static unsigned g_seeds[4] =  {2784565659u, 1491908209u, 3415062841u, 3293636241u};

// parameters stored in constant memory for GPU



void set_params(PARAMS p);
void dump_agents(const char *str, AGENT_DATA *ag);

void free_agnetsCPU(AGENT_DATA *agCPU);
void run_CPU(AGENT_DATA *cv, RESULTS *r);

void initialize_agentsGPU(AGENT_DATA *agGPU);
void free_agentsGPU();
void run_GPU(RESULTS *r);


RESULTS *initialize_results();
void free_results(RESULTS *r);
void display_results(const char *str, RESULTS *r);

#pragma mark CPU & GPU
// random number in the specified range
__device__ __host__ float rand_in_range(unsigned *seeds, unsigned stride, float min, float max)
{
	float r = min + (max-min)*RandUniform(seeds, stride);
	return r;
}

// randomize the state
__device__ __host__ void randomize_state(float *s, unsigned *seeds, unsigned stride)
{
	s[0] = rand_in_range(seeds, stride, MIN_X, MAX_X);
	s[stride] = rand_in_range(seeds, stride, MIN_VEL, MAX_VEL);
}



#pragma mark CPU
RESULTS *initialize_results()
{
	return NULL;
}

// generate random seeds for the sepecified number of agents
unsigned *create_seeds(unsigned num_agents)
{
	unsigned *seeds = (unsigned *)malloc(num_agents * 4 * sizeof(unsigned));
	for (int i = 0; i < num_agents * 4; i++) {
		seeds[i] = RandUniformui(g_seeds, 1);
	}
	return seeds;
}

// create wgts set initially to random values between theta_min and theat_max
float *create_theta(unsigned num_agents, unsigned num_wgts, float theta_min, float theta_max)
{
#ifdef VERBOSE
	printf("create_theta for %d agents and %d weights\n", num_agents, num_wgts);
#endif
	float *theta = (float *)malloc(num_agents * num_wgts * sizeof(float));
	for (int i = 0; i < num_agents * num_wgts; i++) {
		theta[i] = rand_in_range(g_seeds, 1, theta_min, theta_max);
	}
	return theta;
}

// create gradient trace set initially to 0.0f
float *create_W(unsigned num_agents, unsigned num_wgts)
{
#ifdef VERBOSE
	printf("create_theta for %d agents and %d hidden weights\n", num_agents, num_wgts);
#endif
	float *W = (float *)malloc(num_agents * num_wgts * sizeof(float));
	for (int i = 0; i < num_agents * num_wgts; i++) W[i] = 0.0f;
	return W;
}


// create initial random states
float *create_states(unsigned num_agents, unsigned state_size, unsigned *seeds)
{
	float *states = (float *)malloc(num_agents * state_size * sizeof(float));
	for (int i = 0; i < num_agents; i++) {
		randomize_state(states + i, seeds + i, num_agents);
	}
	return states;
}

float *create_Q(unsigned num_agents, unsigned num_actions)
{
	float *Q = (float *)malloc(num_agents * num_actions * sizeof(float));
	for (int i = 0; i < num_agents * num_actions; i++) Q[i] = 0.0f;
	return Q;
}

unsigned *create_actions(unsigned num_agents, unsigned num_actions)
{
	unsigned *actions = (unsigned *)malloc(num_agents * num_actions * sizeof(unsigned));
	for (int i = 0; i < num_agents * num_actions; i++) actions[i] = num_actions; // not valid value
	return actions;
}

AGENT_DATA *initialize_agentsCPU()
{
	AGENT_DATA *ag = (AGENT_DATA *)malloc(sizeof(AGENT_DATA));
	ag->seeds = create_seeds(_p.agents);
	ag->theta = create_theta(_p.agents, _p.hidden_nodes, _p.initial_theta_min, _p.initial_theta_max);
	ag->W = create_W(_p.agents, _p.hidden_nodes);
	ag->s = create_states(_p.agents, _p.state_size, ag->seeds);
	ag->Q = create_Q(_p.agents, _p.num_actions);
	ag->action = create_actions(_p.agents, _p.num_actions);
	return ag;
}

void free_agentsCPU(AGENT_DATA *ag)
{
#ifdef VERBOSE
	printf("freeing agents on CPU...\n");
#endif
	if (ag) {
		if (ag->seeds) free(ag->seeds);
		if (ag->theta) free(ag->theta);
		if (ag->W) free(ag->W);
		if (ag->s) free(ag->s);
		if (ag->Q) free(ag->Q);
		if (ag->action) free(ag->action);
		free(ag);
	}
}

void run_CPU(AGENT_DATA *ag, RESULTS *r)
{
}

#pragma mark -
#pragma mark GPU

void initialize_agentsGPU(AGENT_DATA *agCPU)
{
}

void free_agentsGPU(AGENT_DATA *ag)
{
#ifdef VERBOSE
	printf("freeing agents on GPU...\n");
#endif

}
