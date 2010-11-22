//
//  mcar.cu
//  mcar
//
//  Created by Dwight Bell on 11/20/10.
//  Copyright dbelll 2010. All rights reserved.
//

#include <cuda.h>
#include "cutil.h"
#include <math.h>

#include "main.h"
#include "mcar.h"
#include "cuda_utils.h"
#include "cuda_rand.cu"
#include "misc_utils.h"


// parameters stored in global structure for CPU
static PARAMS _p;

// Initial global seeds used to ensure identical random variables each run on all machines
static unsigned g_seeds[SEEDS_PER_AGENT] =  {2784565659u, 1491908209u, 3415062841u, 3293636241u};


// parameters stored in constant memory for GPU



static float accel[NUM_ACTIONS] = {-ACCEL_FACTOR, 0.0f, ACCEL_FACTOR};

#pragma mark GPU constant memory
__constant__ float dc_accel[NUM_ACTIONS];


#pragma mark CPU & GPU

DUAL_PREFIX float sigmoid(float in)
{
	return 1.0/(1.0 + expf(-in));
}

// calculate the index for the bias weight for hidden node j
DUAL_PREFIX unsigned iHiddenBias(unsigned j, unsigned num_states, unsigned num_actions, unsigned stride)
{
	return j * (1 + num_actions + num_states) * stride;
}


// calculate the index for the bias weight for the output node
DUAL_PREFIX unsigned iOutputBias(unsigned num_states, unsigned num_actions, unsigned num_hidden, unsigned stride)
{
	return num_hidden * (1 + num_states + num_actions) * stride;
}

// Calculate the output of the neural net for specified state and action.
// Hidden node activation values are stored in activation array.
DUAL_PREFIX float calc_Q(float *s, unsigned a, float *theta, unsigned stride, unsigned num_states, unsigned num_actions, unsigned num_hidden, float *activation)
{
	// loop over each hidden node
	for (int j = 0; j < num_hidden; j++) {
		// iBias is the index into theta for the bias weight for the hidden node j
		unsigned iBias = j * (1 + num_actions + num_states) * stride;
		
		// first calculate contribution of the bias for this hidden node
		float in = theta[iBias] * -1.0f;
		
		// next add in the contributions for the state input nodes
		for (int k = 0; k < num_states; k++) {
			in += theta[iBias + (1+k) * stride] * s[k * stride];
		}
		
		// finally, add in the contribution from the selected action
		in += theta[iBias + (1 + num_states + a) * stride];
		
		// apply sigmoid and store in the activation array
		activation[j * stride] = sigmoid(in);

#ifdef DEBUG_CPU
		printf("calc_Q for state (%9.4f, %9.4f) and action %d\n", s[0], s[stride], a);
		printf("input to hidden node %d is %9.4f and activation is %9.4f\n", j, in, activation[j*stride]);
#endif
	}
	
	// Calculate the output Q-value
	// first add in the bias contribution
	unsigned iBias = iOutputBias(num_states, num_actions, num_hidden, stride);
	float result = theta[iBias] * -1.0f;
	
	// loop over the hidden nodes and add in their contribution
	for (int j = 0; j < num_hidden; j++) {
		result += theta[iBias + (1+j) * stride] * activation[j * stride];
	}
#ifdef DEBUG_CPU
		printf("output activation is %9.4f\n", result);
#endif
	return result;
}


DUAL_PREFIX void update_stored_Q(float *Q, float *s, float *theta, unsigned stride, unsigned num_states, unsigned num_actions, unsigned num_hidden, float *activation)
{
	for (int a = 0; a < num_actions; a++) {
		Q[a * stride] = calc_Q(s, a, theta, stride, num_states, num_actions, num_hidden, activation);
	}
}

// Update the weights in the neural net (theta's) using back-propagation of the output error, delta
// Current activation for the hidden layer is pre-calculated in activation
DUAL_PREFIX void update_thetas(unsigned a, float *s, float *theta, float alpha, float delta, unsigned stride, unsigned num_states, unsigned num_actions, unsigned num_hidden, float *activation)
{
	// update the weights from the hidden layer to the output node
	// First the bias
	unsigned iOutBias = iOutputBias(num_states, num_actions, num_hidden, stride);
	theta[iOutBias] += delta * alpha * -1.0f;
	
	// next update each weight from hidden nodes to output node
	for (int j = 0; j < num_hidden; j++) {
		theta[iOutBias + (1+j) * stride] += alpha * activation[j * stride] * delta;
	}
	
	// update weights from input layer to hidden layer for each node in hidden layer
	for (int j = 0; j < num_hidden; j++) {
		// calculate the delta value for this hidden node
		float delta_j = theta[iOutBias + (1+j) * stride] * delta * activation[j * stride] * (1.0f - activation[j * stride]);
		// first update the bias weight
		unsigned iHidBias = iHiddenBias(j, num_states, num_actions, stride);
		theta[iHidBias] += alpha * -1.0f * delta_j;
		
		// update the weights from the state nodes
		for (int k = 0; k < num_states; k++) {
			theta[iHidBias + (k+1) * stride] += alpha * s[k * stride] * delta_j;
		}
		
		// update the weight for the action
		theta[iHidBias + (1 + num_states + a) * stride] += alpha * delta_j;
	}
}

// Calculate the Q value for each action from the given state, storing the values in Q
// Return the action with the highest Q value
DUAL_PREFIX unsigned best_action(float *s, float *theta, float *Q, unsigned stride, unsigned num_states, unsigned num_actions, unsigned num_hidden, float *activation)
{
	// calculate Q value for each action
	Q[0] = calc_Q(s, 0, theta, stride, num_states, num_actions, num_hidden, activation);
	unsigned best_action = 0;
	float bestQ = Q[0];
	for (int a = 1; a < num_actions; a++) {
		Q[a*stride] = calc_Q(s, a, theta, stride, num_states, num_actions, num_hidden, activation);
		if (Q[a*stride] > bestQ) {
			bestQ = Q[a*stride];
			best_action = a;
		}
	}
	return best_action;
}

// choose action from current state, storing Q values for each possible action in Q, and 
DUAL_PREFIX unsigned choose_action(float *s, float *theta, float epsilon, unsigned stride, float *Q, unsigned num_states, unsigned num_actions, unsigned num_hidden, float *activation, unsigned *seeds)
{
	// always calcualte the best action and store all the Q values for each action
	unsigned a = best_action(s, theta, Q, stride, num_states, num_actions, num_hidden, activation);
	if (epsilon > 0.0f && RandUniform(seeds, stride) < epsilon){
		// choose random action
		float r = RandUniform(seeds, stride);
		a = r * num_actions;
	}
	return a;
}

DUAL_PREFIX unsigned terminal_state(float *s)
{
	return s[0] >= MAX_X;
}

// take an action from the current state, s, returning the reward and saving new state in s_prime
// Note, s & s_prime may be the same location.
DUAL_PREFIX float take_action(unsigned a, float *s, float *s_prime, unsigned stride, float *accel)
{
	// Forumlation of mountain car problem is from Sutton & Barto, 
	// "Reinforcement Learning, An Introduction"
	
#ifdef DEBUG_CPU
	printf("take_action %s from state (%9.4f, %9.4f)\n", a == 0 ? "LEFT" : (a == 1 ? "NONE" : "RIGHT"), s[0], s[stride]);
#endif

	// normal reward is -1.0f per time step
	float reward = -1.0f;
	
	// update velocity and limit it to within bounds	
	s_prime[stride] = s[stride] + accel[a] + GRAVITY_FACTOR * cosf(GRAVITY_X_SCALE * s[0]);
#ifdef DEBUG_CPU
	printf("accel is %9.6f from force and %9.6f from gravity resulting in new velocity of %9.6f\n", accel[a], GRAVITY_FACTOR * cosf(GRAVITY_X_SCALE * s[0]), s_prime[stride]);
#endif
	if (s_prime[stride] < MIN_VEL) s_prime[stride] = MIN_VEL;
	if (s_prime[stride] > MAX_VEL) s_prime[stride] = MAX_VEL;
	
	// update position and test for success and limit with minimum bound
	s_prime[0] = s[0] + s_prime[stride];
	if (s_prime[0] >= MAX_X) reward = 0.0f;
	if (s_prime[0] <= MIN_X) { s_prime[0] = MIN_X; s_prime[stride] = 0.0f;}
#ifdef DEBUG_CPU
	printf("new state is (%9.4f, %9.4f)\n", s_prime[0], s_prime[stride]);
#endif
	return reward;
}



// random number in the specified range
DUAL_PREFIX float rand_in_range(unsigned *seeds, unsigned stride, float min, float max)
{
	float r = min + (max-min)*RandUniform(seeds, stride);
	return r;
}

// randomize the position and velocity uniformly over their range
DUAL_PREFIX void randomize_state(float *s, unsigned *seeds, unsigned stride)
{
	s[0] = rand_in_range(seeds, stride, MIN_X, MAX_X);
	s[stride] = rand_in_range(seeds, stride, MIN_VEL, MAX_VEL);
}


#pragma mark -
#pragma mark CPU

void set_params(PARAMS p){ _p = p;}

// dump agent data to stdout
// uses parameter values in _p
void dump_agent(AGENT_DATA *ag, unsigned agent)
{
	printf("[agent %d]: ", agent);
	printf("   seeds = %u, %u, %u, %u\n", ag->seeds[agent], ag->seeds[agent + _p.agents], 
									   ag->seeds[agent + 2*_p.agents], ag->seeds[agent + 3*_p.agents]);
	printf("  FROM          TO       THETA       W  \n");
	unsigned i = agent;
	for (int h = 0; h < _p.hidden_nodes; h++) {
		printf("    bias --> hidden%2d %9.4f %9.4f\n", h, ag->theta[i], ag->W[i]); i += _p.agents;
		printf("      x  --> hidden%2d %9.4f %9.4f\n", h, ag->theta[i], ag->W[i]); i += _p.agents;
		printf("      x' --> hidden%2d %9.4f %9.4f\n", h, ag->theta[i], ag->W[i]); i += _p.agents;
		printf("    LEFT --> hidden%2d %9.4f %9.4f\n", h, ag->theta[i], ag->W[i]); i += _p.agents;
		printf("    NONE --> hidden%2d %9.4f %9.4f\n", h, ag->theta[i], ag->W[i]); i += _p.agents;
		printf("   RIGHT --> hidden%2d %9.4f %9.4f\n", h, ag->theta[i], ag->W[i]); i += _p.agents;
	}
	printf("    bias --> output   %9.4f %9.4f\n", ag->theta[i], ag->W[i]); i += _p.agents;
	for (int h = 0; h < _p.hidden_nodes; h++) {
		printf("hidden%2d --> output   %9.4f %9.4f\n", h, ag->theta[i], ag->W[i]); i += _p.agents;
	}

	printf("\nCurrent State: x = %7.4f   x' = %7.4f\n", ag->s[agent], ag->s[agent + _p.agents]);
	printf("ACTION  Q-value\n");
	for (int action = 0; action < _p.num_actions; action++) {
		(action == ag->action[agent]) ? printf("-->") : printf("   ");
		printf("%3d  %9.6f\n", action, ag->Q[agent + action * _p.agents]);
	}
	printf("HIDDEN NODE    ACTIVATION\n");
	for (int j = 0; j < _p.hidden_nodes; j++) {
		printf("   %3d      %9.4f\n", j, ag->activation[agent + j * _p.agents]);
	}
	printf("\n");
}

// print message and dump all agent data
void dump_agents(const char *str, AGENT_DATA *ag)
{
	printf("%s\n", str);
	for (int agent = 0; agent < _p.agents; agent++) {
		dump_agent(ag, agent);
	}
}

void dump_one_agent(const char *str, AGENT_DATA *ag)
{
	printf("%s\n", str);
	dump_agent(ag, 0);
}


RESULTS *initialize_results()
{
	return NULL;
}

// generate random seeds for the sepecified number of agents
unsigned *create_seeds(unsigned num_agents)
{
#ifdef VERBOSE
	printf("create_seeds for %d agents\n", num_agents);
#endif
	unsigned *seeds = (unsigned *)malloc(num_agents * SEEDS_PER_AGENT * sizeof(unsigned));
	for (int i = 0; i < num_agents * SEEDS_PER_AGENT; i++) {
		seeds[i] = RandUniformui(g_seeds, 1);
	}
	return seeds;
}

// create wgts set initially to random values between theta_min and theat_max
float *create_theta(unsigned num_agents, unsigned num_wgts, float theta_min, float theta_max)
{
#ifdef VERBOSE
	printf("create_theta for %d agents and %d weights in range %6.4f to %6.4f\n", num_agents, num_wgts, theta_min, theta_max);
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
	printf("create_W for %d agents and %d weights\n", num_agents, num_wgts);
#endif
	float *W = (float *)malloc(num_agents * num_wgts * sizeof(float));
	for (int i = 0; i < num_agents * num_wgts; i++) W[i] = 0.0f;
	return W;
}


// create initial random states
float *create_states(unsigned num_agents, unsigned state_size, unsigned *seeds)
{
#ifdef VERBOSE
	printf("create_states for %d agents and state size of %d\n", num_agents, state_size);
#endif
	float *states = (float *)malloc(num_agents * state_size * sizeof(float));
	for (int i = 0; i < num_agents * state_size; i++) states[i] = 0.0f;
	return states;
}

float *create_Q(unsigned num_agents, unsigned num_actions)
{
#ifdef VERBOSE
	printf("create_Q for %d agents and %d actions\n", num_agents, num_actions);
#endif
	float *Q = (float *)malloc(num_agents * num_actions * sizeof(float));
	for (int i = 0; i < num_agents * num_actions; i++) Q[i] = 0.0f;
	return Q;
}

unsigned *create_actions(unsigned num_agents, unsigned num_actions)
{
#ifdef VERBOSE
	printf("create_actions for %d agents\n", num_agents);
#endif
	unsigned *actions = (unsigned *)malloc(num_agents * num_actions * sizeof(unsigned));
	for (int i = 0; i < num_agents * num_actions; i++) actions[i] = num_actions; // not valid value
	return actions;
}

float *create_activation(unsigned num_agents, unsigned num_hidden)
{
#ifdef VERBOSE
	printf("create_activation for %d agents wiht %d hidden nodes\n", num_agents, num_hidden);
#endif
	float *activation = (float *)malloc(num_agents * (num_hidden) * sizeof(float));
	for (int i = 0; i < num_agents * num_hidden; i++) activation[i] = 0.0f;
	return activation;
}
AGENT_DATA *initialize_agentsCPU()
{
#ifdef VERBOSE
	printf("initializing agents on CPU...\n");
#endif
	AGENT_DATA *ag = (AGENT_DATA *)malloc(sizeof(AGENT_DATA));
	ag->seeds = create_seeds(_p.agents);
	ag->theta = create_theta(_p.agents, _p.num_wgts, _p.initial_theta_min, _p.initial_theta_max);
	ag->W = create_W(_p.agents, _p.num_wgts);
	ag->s = create_states(_p.agents, _p.state_size, ag->seeds);
	ag->Q = create_Q(_p.agents, _p.num_actions);
	ag->action = create_actions(_p.agents, _p.num_actions);
	ag->activation = create_activation(_p.agents, _p.hidden_nodes);
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


void learning_session(AGENT_DATA *ag)
{
	// for each agent
	for (int agent = 0; agent < _p.agents; agent++) {
		// for each time step
		for (int t = 0; t < _p.chunk_interval; t++) {
			//accumulate_gradient(...
			float reward = take_action(ag->action[agent], ag->s + agent, ag->s + agent, _p.agents, accel);
			unsigned success = terminal_state(ag->s + agent);
			if (success) randomize_state(ag->s + agent, ag->seeds + agent, _p.agents);
			float Q_curr = ag->Q[agent + ag->action[agent] * _p.agents];
			ag->action[agent] = choose_action(ag->s + agent, ag->theta + agent, _p.epsilon, _p.agents, ag->Q + agent, _p.state_size, _p.num_actions, _p.hidden_nodes, ag->activation + agent, ag->seeds + agent);
			float Q_next = ag->Q[agent + ag->action[agent] * _p.agents];
			float delta = reward + _p.gamma*Q_next - Q_curr;
			update_thetas(ag->action[agent], ag->s + agent, ag->theta + agent, _p.alpha, delta, _p.agents, _p.state_size, _p.num_actions, _p.hidden_nodes, ag->activation + agent);
//			if (success) reset_trace(...
			update_stored_Q(ag->Q + agent, ag->s + agent, ag->theta + agent, _p.agents, _p.state_size, _p.num_actions, _p.hidden_nodes, ag->activation + agent);
//			update_trace(...
		}
	}
}

// share is where the best agents will be selected and duplicated
void share(AGENT_DATA *ag)
{
		
}

void randomize_all_states(AGENT_DATA *ag)
{
	// randomize state for all agents, deterine first action and 
	for (int agent = 0; agent < _p.agents; agent++) {
		randomize_state(ag->s + agent, ag->seeds + agent, _p.agents);
		ag->action[agent] = choose_action(ag->s + agent, ag->theta + agent, _p.epsilon, _p.agents, ag->Q + agent, _p.state_size, _p.num_actions, _p.hidden_nodes, ag->activation + agent, ag->seeds + agent);
		// force activation values to be recalculated for the chosen action
		calc_Q(ag->s + agent, ag->action[agent], ag->theta + agent, _p.agents, _p.state_size, _p.num_actions, _p.hidden_nodes, ag->activation + agent);
		// update_trace(...
	}
}

void run_CPU(AGENT_DATA *ag, RESULTS *r)
{
#ifdef VERBOSE
	printf("\n==============================================\nrunning on CPU...\n");
#endif
#ifdef DUMP_INITIAL_AGENTS
	dump_agents("Initial agents on CPU", ag);
#endif
	unsigned timer;
	CREATE_TIMER(&timer);
	START_TIMER(timer);

	timing_feedback_header(_p.num_chunks);
	randomize_all_states(ag);
	
	dump_agents("after initial randomization", ag);
	
	for (int i = 0; i < _p.num_chunks; i++) {
		timing_feedback_dot(i);
		learning_session(ag);
		
		dump_agents("after some learning...", ag);
		
		if ((_p.agent_group_size > 1) && 0 == ((i+1) % _p.chunks_per_share)) {
			share(ag);
		}
	}

	STOP_TIMER(timer, "run on CPU");
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
