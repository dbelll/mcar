#ifndef __MCAR_H__
#define __MCAR_H__

//
//  mcar.h
//  mcar
//
//  Created by Dwight Bell on 11/20/10.
//  Copyright dbelll 2010. All rights reserved.
//

#pragma mark -
#pragma mark Problem Constants

#define MIN_X -1.2f
#define MAX_X 0.5f
#define MIN_VEL -0.07f
#define MAX_VEL 0.07f

#define ACCEL_FACTOR 0.001f

// GRAVITY acceleration = GRAVITY_FACTOR * cos(GRAVITY_X_SCALE * x)
#define GRAVITY_FACTOR -0.0025
#define GRAVITY_X_SCALE 3.0f

#define DEFAULT_ALPHA 0.20f
#define DEFAULT_EPSILON 0.00f
#define DEFAULT_GAMMA 0.90f
#define DEFAULT_LAMBDA 0.70f

#define DEFAULT_HIDDEN_NODES 1

#define STATE_SIZE 2
#define NUM_ACTIONS 3
#define SEEDS_PER_AGENT 4


/*
	Parameters are stored in a large structure, including constant values
	that are calculated based on the parameters.
*/

typedef struct{
	unsigned trials;			// number of times to repeat the entire expderiment
	unsigned time_steps;		// number of time steps in one trial
	unsigned agent_group_size;	// number of agents in a group that will work toward one solution
	
	unsigned sharing_interval;	// number of time steps agents work independently before sharing
	
	unsigned agents;			// total number of agents = agent_group_size * trials
	unsigned num_sharing_intervals;	// number of sharing intervals = timp_steps / sharing_interval
	
	float epsilon;				// exploration factor
	float gamma;				// discount factor
	float lambda;				// eligibility trace decay factor
	float alpha;				// learning rate
	
	float initial_theta_min;	// range for initial random weights
	float initial_theta_max;
	
	unsigned run_on_CPU;		// flag indicating to run on CPU
	unsigned run_on_GPU;		// flag indicating to run on GPU
	unsigned no_print;			// flag to suppress print-out
	
	unsigned test_interval;		// number of time steps between testing
	unsigned test_reps;			// number of reps in each test
	unsigned num_tests;			// calculated = time_steps / test_interval
	
	unsigned chunk_interval;	// the number of time steps in the smallest value of
								// sharing_interval, test_interval, restart_interval
	unsigned num_chunks;		// calculated = time_steps / chunk_interval
	unsigned chunks_per_share;	// calculated = sharing_interval / chunk_interval
	unsigned chunks_per_test;	// calculated = test_interval / chunk_interval
	
	unsigned hidden_nodes;	// number of hidden nodes in the neural net used to calculate Q(s,a)
	unsigned num_wgts;		// total number of weights = 7*hidden_nodes + 1
	unsigned num_actions;	// 3
	unsigned state_size;	// 2 (x and x')
	unsigned input_nodes;	// state_size + num_actions
} PARAMS;


/*
 *	The AGENT_DATA structure holds pointers to data related to all the agents.
 */
typedef struct{
	unsigned *seeds;	// seeds for random number generator
	float *theta;		// weights for the neural net organized as follows:
						// bias -> hidden[0]
						// x -> hidden[0]
						// x' -> hidden[0]
						// a[0] (negative acceleration) -> hidden[0]
						// a[1] (no acceleration) -> hidden[1]
						// a[2] (positive acceleration) -> hidden[2]
						// ... repeat for other hidden ...
						// ... total of 6 x num_hidden ...
						// bias -> output
						// hidden[0] -> output
						// ... repeat for other hidden ...
						// ... total of 1 = num_hidden ...
						// ... grand total of 7 * num_hidden + 1 ...
						//
	float *W;			// sum of lambda * gamma * gradient of Q for each weight in neural net
	float *s;			// current state x and x'
	float *activation;	// activation values for hidden nodes - must be stored so they can
						// be used during back propagation
	float *Q;			// Q values for each action, filled when determining best action
	unsigned *action;	// temp storage for action to be taken at the next action
} AGENT_DATA;		// may hold either host or device pointers

typedef struct{
	float *avg_steps;
} RESULTS;


void set_params(PARAMS p);
void dump_agents(const char *str, AGENT_DATA *ag);

AGENT_DATA *initialize_agentsCPU();
void free_agnetsCPU(AGENT_DATA *agCPU);
void run_CPU(AGENT_DATA *cv, RESULTS *r);

void initialize_agentsGPU(AGENT_DATA *agCPU);
void free_agentsGPU();
void run_GPU(RESULTS *r);


RESULTS *initialize_results();
void free_results(RESULTS *r);
void display_results(const char *str, RESULTS *r);


#endif
