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

#include "cuda_utils.h"
#include "./common/inc/cutil.h"
#include "mcar.h"

// print out information on using this program
void display_help()
{
	printf("bandit parameters:\n");
	printf("  --TRIALS              number of trials for averaging reults\n");
	printf("  --TIME_STEPS          total number of time steps for each trial\n");
	printf("  --AGENT_GROUP_SIZE    size of agent groups that will communicate\n");
	printf("  --SHARING_INTERVAL    number of time steps between agent communication\n");
	
	printf("  --ALPHA               float value for alpha, the learning rate parameter\n");
	printf("  --EPSILON             float value for epsilon, the exploration parameter\n");
	printf("  --GAMMA               float value for gamma, the discount factor\n");
	printf("  --LAMBDA              float value for lambda, the trace decay factor\n");
	
	printf("  --INIT_THETA_MIN		minimum of range of possible initial theta values\n");
	printf("  --INIT_THETA_MAX		maximum of range of possible initial theta values\n");
	
	printf("  --RUN_ON_GPU          1 = run on GPU, 0 = do not run on GPU\n");
	printf("  --RUN_ON_CPU          1 = run on CPU, 0 = do not run on CPU\n");
	printf("  --NO_PRINT			flag to suppress printing out results (only timing values printed)\n");
	
	printf("  --TEST_INTERVAL       time steps between testing of agent's learning ability\n");
	printf("  --TEST_REPS			duration of test in time steps\n");
	printf("  --RESTART_INTERVAL    time steps between random restarts\n");
	
	printf("  --HELP                print this help message\n");
	printf("default values will be used for any parameters not on command line\n");
}


// read parameters from command line (or use default values) and print the header for this run
PARAMS read_params(int argc, const char **argv)
{
#ifdef VERBOSE
	printf("reading parameters...\n");
#endif
	PARAMS p;
	if (argc == 1 || PARAM_PRESENT("HELP")) { display_help(); exit(1); }
	
	p.trials = GET_PARAM("TRIALS", 1024);
	p.time_steps = GET_PARAM("TIME_STEPS", 64);
	p.agent_group_size = GET_PARAM("AGENT_GROUP_SIZE", 1);
	p.agents = p.trials * p.agent_group_size;
	p.sharing_interval = GET_PARAM("SHARING_INTERVAL", p.time_steps);
	
	// set sharing interval to total time steps if only one agent
	if (p.agents == 1) p.sharing_interval = p.time_steps;
	
	// Total time steps must be an integer number of sharing intervals
	if (p.agent_group_size > 1 && 0 != (p.time_steps % p.sharing_interval)){
		printf("Inconsistent arguments: TIME_STEPS=%d, SHARING_INTERVAL=%d\n", 
			   p.time_steps, p.sharing_interval);
		exit(1);
	}
	p.num_sharing_intervals = p.time_steps / p.sharing_interval;
	
	p.initial_theta_min = GET_PARAMF("INIT_THETA_MIN", 0.0f);
	p.initial_theta_max = GET_PARAMF("INIT_THETA_MAX", 1.0f);

	p.alpha = GET_PARAMF("ALPHA", DEFAULT_ALPHA);
	p.epsilon = GET_PARAMF("EPSILON", DEFAULT_EPSILON);
	p.gamma = GET_PARAMF("GAMMA", DEFAULT_GAMMA);
	p.lambda = GET_PARAMF("LAMBDA", DEFAULT_LAMBDA);
	
	p.run_on_CPU = GET_PARAM("RUN_ON_CPU", 1);
	p.run_on_GPU = GET_PARAM("RUN_ON_GPU", 1);
	p.no_print = PARAM_PRESENT("NO_PRINT");
	
	p.test_interval = GET_PARAM("TEST_INTERVAL", p.time_steps);
	p.test_reps = GET_PARAM("TEST_REPS", p.test_interval);
	p.num_tests = p.time_steps / p.test_interval;
	if (p.test_interval > p.time_steps){
		printf("Inconsistent arguments: TEST_INTERVAL = %d is greater than TIME_STEPS = %d.\n",
			   p.test_interval, p.time_steps);
		exit(1);
	}
	
	// calculate chunk_interval as smallest of other intervals, or 
	p.chunk_interval = p.test_interval;
	if(p.chunk_interval > p.sharing_interval) p.chunk_interval = p.sharing_interval;
	

	// use value from command line if present
	p.chunk_interval = GET_PARAM("CHUNK_INTERVAL", p.chunk_interval);
	if (p.chunk_interval > p.test_interval ||
		p.chunk_interval > p.sharing_interval) {
		printf("Inconsistent arguments: CHUNK_INTERVAL = %d but must be <= all other intervals and evenly divide them.\n",
			   p.chunk_interval);
		exit(1);
	}

	if (0 != (p.time_steps % p.chunk_interval)) {
		printf("Inconsistent arguments: TIME_STEPS=%d, but time chunks are calculated to be %d\n", p.time_steps, 
			   p.chunk_interval);
		exit(1);
	}
	
	p.num_chunks = p.time_steps / p.chunk_interval;
	
	// test interval must be a positive integer times the chunk interval
	if (p.chunk_interval > p.test_interval || 0 != (p.test_interval % p.chunk_interval)) {
		printf("Inconsistent arguments: TEST_INTERVAL=%d, but time chunks are calculated as %d\n", p.test_interval, 
			   p.chunk_interval);
		exit(1);
	}
	
	// sharing interval must be a positive integer times the chunk interval
	if (p.chunk_interval > p.sharing_interval || 0 != (p.sharing_interval % p.chunk_interval)) {
		printf("Inconsistent arguments: SHARING_INTERVAL=%d, but time chunks are calculated as %d\n", 
			   p.sharing_interval, p.chunk_interval);
		exit(1);
	}
	
	p.chunks_per_test = p.test_interval / p.chunk_interval;
	p.chunks_per_share = p.sharing_interval / p.chunk_interval;
		
	p.state_size = 2;	// x and x'
	p.num_actions = 3;	// force left, no force, force right
	
	printf("[POLE][TRIALS%7d][TIME_STEPS%7d][SHARING_INTERVAL%7d][AGENT_GROUP_SIZE%7d][ALPHA%7.4f]"
		   "[EPSILON%7.4f][GAMMA%7.4f][LAMBDA%7.4f][TEST_INTERVAL%7d][TEST_REPS%7d][CHUNK_INTERVAL%7d]\n", 
		   p.trials, p.time_steps, p.sharing_interval, p.agent_group_size, p.alpha, 
		   p.epsilon, p.gamma, p.lambda, p.test_interval, p.test_reps, p.chunk_interval);
	
	return p;
}

int main(int argc, const char **argv)
{
	PARAMS p = read_params(argc, argv);

	// Initialize agents on CPU and GPU
	AGENT_DATA *agCPU = initialize_agentsCPU();
	if (p.run_on_GPU) initialize_agentsGPU(agCPU);
	
	RESULTS *rCPU = NULL;
	RESULTS *rGPU = NULL;
	
	if (p.run_on_CPU) {
		rCPU = initialize_results();
		run_CPU(agCPU, rCPU);
	}
	
	if (p.run_on_GPU) {
		rGPU = initialize_results();
	}
	
	
	return 0;
}
