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

#define LEARN_BLOCK_SIZE 64				// used for learning kernel
#define TEST3_BLOCK_SIZE 256			// used for test_kernel3 which runs the competition
#define CALC_QUALITY_BLOCK_SIZE_PRELIM 512		// used for calc_all_quality kernel
#define SHARE_BEST_BLOCK_SIZE 512		// used for share_best_kernel

#define MIN_X (-1.2f)
#define MAX_X 0.5f
#define MIN_VEL (-0.07f)
#define MAX_VEL 0.07f

// parameters for calculating agent quality
// The fine division is used for recording quality in AGENT_DATA and for testing if competition
// winner is better than current best.
#define NUM_X_DIV 64
#define NUM_VEL_DIV 64
#define DIV_X ((MAX_X - MIN_X) / NUM_X_DIV)
#define DIV_VEL ((MAX_VEL - MIN_VEL)/NUM_VEL_DIV);
//#define DIV_X 0.020
//#define DIV_VEL 0.002
//#define NUM_X_DIV ((unsigned)(1.5f  + (MAX_X - MIN_X) / DIV_X))
//#define NUM_VEL_DIV ((unsigned)(1.5f + (MAX_VEL - MIN_VEL) / DIV_VEL))

#define NUM_TOT_DIV (NUM_X_DIV * NUM_VEL_DIV)

// The crude divs are  used when testing all agents when compete = no
#define CRUDE_NUM_X_DIV 12
#define CRUDE_NUM_VEL_DIV 12
#define CRUDE_DIV_X ((MAX_X - MIN_X) / CRUDE_NUM_X_DIV)
#define CRUDE_DIV_VEL ((MAX_VEL - MIN_VEL)/CRUDE_NUM_VEL_DIV)
//#define CRUDE_DIV_X 0.100
//#define CRUDE_DIV_VEL 0.0100
//#define CRUDE_NUM_X_DIV ((unsigned)(1.5f  + (MAX_X - MIN_X) / CRUDE_DIV_X))
//#define CRUDE_NUM_VEL_DIV ((unsigned)(1.5f + (MAX_VEL - MIN_VEL) / CRUDE_DIV_VEL))
#define CRUDE_NUM_TOT_DIV (CRUDE_NUM_X_DIV * CRUDE_NUM_VEL_DIV)

#if CRUDE_NUM_TOT_DIV < CALC_QUALITY_BLOCK_SIZE_PRELIM
#define CALC_QUALITY_BLOCK_SIZE CRUDE_NUM_TOT_DIV
#else
#define CALC_QUALITY_BLOCK_SIZE CALC_QUALITY_BLOCK_SIZE_PRELIM
#endif


#define MAX_STEPS_FOR_QUALITY 300		// used when calc'ing quality for all agents (no compete)
#define FINAL_QUALITY_MAX_STEPS 2000		// used for recording winner quality on the AGENT_DATA, and
										// to test if the competition winner is better than current best

#define ACCEL_FACTOR 0.001f

// GRAVITY acceleration = GRAVITY_FACTOR * cos(GRAVITY_X_SCALE * x)
#define GRAVITY_FACTOR -0.0025
#define GRAVITY_X_SCALE 3.0f

#define DEFAULT_ALPHA 0.20f
#define DEFAULT_EPSILON 0.00f
#define DEFAULT_GAMMA 0.90f
#define DEFAULT_LAMBDA 0.70f

#define DEFAULT_SHARE_BEST_PCT 0.50f
#define DEFAULT_SHARE_ALWAYS_PCT 0.00f

#define DEFAULT_TEST_REPS 10
#define DEFAULT_TEST_MAX 1000
#define MAX_FITNESS 9999 * CRUDE_NUM_TOT_DIV

#define DEFAULT_HIDDEN_NODES 1

#define STATE_SIZE 2
#define NUM_ACTIONS 3
#define SEEDS_PER_AGENT 4
#define NUM_HIDDEN 1
#define NUM_WGTS (NUM_ACTIONS * ((1 + STATE_SIZE) * NUM_HIDDEN + (1 + NUM_HIDDEN)))

#if NUM_WGTS > CALC_QUALITY_BLOCK_SIZE
#undef NUM_WGTS
#endif


/*
	Parameters are stored in a large structure, including constant values
	that are calculated based on the parameters.
*/

typedef struct{
	unsigned trials;			// number of times to repeat the entire expderiment
	unsigned time_steps;		// number of time steps in one trial
	unsigned agent_group_size;	// number of agents in a group that will work toward one solution
	
//	unsigned sharing_interval;	// number of time steps agents work independently before sharing
	float share_best_pct;		// the probability of beling replaced by the best agent during
								// sharing, applies only when there is a new best agent
	float share_always_pct;		// the probability of beling replaced by the best agent during
								// sharing, applies when the best agent has not changed
	unsigned share_compete;		// flag to indicate that competition is used for sharing
	unsigned share_fitness;		// flag to indicate that fitness is used for sharing
//	unsigned share_always;		// flag indicates that the best agent is always shared with losers, even when it has not changed from the previous best agent.
	float copy_alpha_multiplier;	// copied agent's alpha is the normal alpha times this factor
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
	unsigned dump_all_winners;	// flag to print out all agents that win competition
	unsigned dump_all_new_best;	// flag to print out all new best agents
	unsigned dump_best;			// flag to print out the best agent at the end
	unsigned dump_updates;		// print update information when a new best agent is found
	
	unsigned test_interval;		// number of time steps between testing
	unsigned test_reps;			// number of repitions of the test
	unsigned test_max;			// maximum time steps in the test
	unsigned num_tests;			// calculated = time_steps / test_interval

	unsigned restart_interval;	// interval for restarting the agent at a random state
	
	unsigned chunk_interval;	// the number of time steps in the smallest value of
								// sharing_interval, test_interval, restart_interval
	unsigned num_chunks;		// calculated = time_steps / chunk_interval
//	unsigned chunks_per_share;	// calculated = sharing_interval / chunk_interval
	unsigned chunks_per_test;	// calculated = test_interval / chunk_interval
	unsigned chunks_per_restart;	// calculated = restart_interval / chunk_interval
	
//	unsigned hidden_nodes;	// number of hidden nodes in the neural net used to calculate Q(s,a)
//	unsigned num_wgts;		// total number of weights for each action = 12*hidden_nodes + 3
//	unsigned num_actions;	// 3
//	unsigned state_size;	// 2 (x and x')
//
//	unsigned num_states;	// alternate names
	unsigned stride;
//	unsigned num_hidden;
	
	unsigned iActionStart[NUM_ACTIONS];
	unsigned offsetToOutputBias;
	
} PARAMS;


/*
 *	The AGENT_DATA structure holds pointers to data related to all the agents.
 */
typedef struct{
	unsigned *seeds;	// seeds for random number generator (num_agents * 4)
	float *theta;		// weights for the neural, net organized as follows:
						// LEFT Nerual Net
						// bias -> hidden[0]
						// x -> LEFT_NN hidden[0]
						// x' -> LEFT_NN hidden[0]
						// ... repeat for other hidden ...
						// ... total of 3 x num_hidden ...
						// bias -> output
						// LEFT_NN hidden[0] -> output
						// ... repeat for other hidden ...
						// ... total of 1 + num_hidden ...
						// ... grand total of 4 * num_hidden + 1 ...
						//
						// Repeat for NONE Neural Net and RIGHT Neural Net
						// Grand total of num_actions * (4 * num_hidden + 1) 
						// for mcar, this equals 12*num_hidden + 3 weights
						//
	float *W;			// sum of lambda * gamma * gradient of Q for each weight in neural net
						// same size as theta
	float *s;			// current state x and x'
	float *activation;	// activation values for hidden nodes (num_hidden) 
						// must be stored so they can be used during back propagation
	unsigned *action;	// temp storage for action to be taken at the next action
	float *fitness;		// fitness value from last test
	float *alpha;		// personal alpha value
	unsigned *alphaOn;	// flag to indicate alpha should be turned on
} AGENT_DATA;		// may hold either host or device pointers

typedef struct{
	float *avg_fitness;		// average fitness of all agents
	float *best_fitness;	// best fitness value
	unsigned *best_agent;	// index of agent with best fitness value
} RESULTS;

typedef struct{
	unsigned time_step;	// timestep of taking over as new
	unsigned agent;		// agent number
	float fitness;		// agent fitness
	float time;			// amount of time for this episode
//	float *theta;		// copy of the theta values at that time  //**TODO implement later
} GPU_RESULTS;			// records information when a new best agent is found.

void set_seed(unsigned seed);
void set_params(PARAMS p);
void dump_agents(const char *str, AGENT_DATA *ag);
void dump_agent_pointers(const char *str, AGENT_DATA *ag);

AGENT_DATA *initialize_agentsCPU();
void free_agentsCPU(AGENT_DATA *agCPU);
void run_CPU(AGENT_DATA *cv);

AGENT_DATA *initialize_agentsGPU(AGENT_DATA *agCPU);
void dump_agentsGPU(const char *str, AGENT_DATA *agGPU, unsigned crude);
void dump_one_agentGPU(const char *str, AGENT_DATA *agGPU, unsigned ag, unsigned crude);
void free_agentsGPU(AGENT_DATA *agGPU);
void run_GPU(AGENT_DATA *ag);


//RESULTS *initialize_results();
void free_results(RESULTS *r);
void display_results(const char *str, RESULTS *r);


#endif
