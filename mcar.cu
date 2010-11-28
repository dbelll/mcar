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
#include <assert.h>

#include "main.h"
#include "mcar.h"
#include "cuda_utils.h"
#include "cuda_rand.cu"
#include "misc_utils.h"
#include "reduction.h"

// parameters stored in global structure for CPU
static PARAMS _p;

// Initial global seeds used to ensure identical random variables each run on all machines
static unsigned g_seeds[SEEDS_PER_AGENT] =  {2784565659u, 1491908209u, 3415062841u, 3293636241u};

static float accel[NUM_ACTIONS] = {-ACCEL_FACTOR, 0.0f, ACCEL_FACTOR};


#pragma mark GPU constant memory
__constant__ float dc_accel[NUM_ACTIONS];

__constant__ PARAMS dc_p;
__constant__ AGENT_DATA dc_ag;

// paramaters are stored in constant memory on the device
//__constant__ unsigned dc_agents;
//__constant__ unsigned dc_agent_group_size;
//__constant__ unsigned dc_time_steps;
//
//__constant__ float dc_epsilon;
//__constant__ float dc_gamma;
//__constant__ float dc_lambda;
//__constant__ float dc_alpha;
//
//__constant__ unsigned dc_num_actions;
//__constant__ unsigned dc_num_hidden;
//
//__constant__ unsigned dc_test_interval;
//__constant__ unsigned dc_test_reps;
//__constant__ unsigned dc_test_max;

// fixed pointers are stored in constant memory on the device
//__constant__ unsigned *dc_seeds;
//__constant__ float *dc_theta;
//__constant__ float *dc_W;
//__constant__ float *dc_s;
//__constant__ unsigned *dc_action;
//__constant__ float *fitness;


const char * string_for_action(unsigned a)
{
	return (a == 0) ? "LEFT" : ((a == 1) ? "NONE" : "RIGHT");
}



#pragma mark -
#pragma mark CPU & GPU


DUAL_PREFIX float sigmoid(float in)
{
	return 1.0f/(1.0f + expf(-in));
}

DUAL_PREFIX unsigned iActionStart(unsigned a, unsigned stride, unsigned num_hidden)
{
	unsigned i = (a * ((STATE_SIZE + 2) * num_hidden + 1)) * stride;
//	printf("iActionStart for action %d is %d\n", a, i);
	return i;
}

// calculate the offset from the start of the weights for the bias weight for hidden node j
DUAL_PREFIX unsigned offsetToHiddenBias(unsigned j, unsigned stride, unsigned num_hidden)
{
	unsigned i = j*(1 + STATE_SIZE) * stride;
//	printf("offsetToHiddenBias for hidden node %d is %d\n", j, i);
	return i;
}


// calculate the index for the bias weight for the output node
DUAL_PREFIX unsigned offsetToOutputBias(unsigned stride, unsigned num_hidden)
{
	unsigned i = num_hidden*(1 + STATE_SIZE) * stride;
//	printf("offsetToOutputBias is %d\n", i);
	return i;
}

// Calculate the output of the neural net for specified state and action.
// Hidden node activation values are stored in activation array and the output Q value is returned.
DUAL_PREFIX float calc_Q(float *s, unsigned a, float *theta, unsigned stride, unsigned num_hidden, float *activation)
{
	// adjust theta to point to beginning of this action's weights
	theta += iActionStart(a, stride, num_hidden);
	
	unsigned iOutputBias = offsetToOutputBias(stride, num_hidden);
	
	float result = 0.0f;

	// loop over each hidden node
	for (int j = 0; j < num_hidden; j++) {
		// iBias is the index into theta for the bias weight for the hidden node j
		unsigned iBias = offsetToHiddenBias(j, stride, num_hidden);
		
		// first calculate contribution of the bias for this hidden node
		float in = theta[iBias] * -1.0f;
		
		// next add in the contributions for the state input nodes
		for (int k = 0; k < STATE_SIZE; k++) {
			in += theta[iBias + (1+k) * stride] * s[k * stride];
		}
		
		activation[j * stride] = sigmoid(in);
		result += theta[iOutputBias + (1+j) * stride] * activation[j*stride];

#ifdef DEBUG_CALC_Q
		printf("calc_Q for state (%9.4f, %9.4f) and action %d ... ", s[0], s[stride], a);
//		printf("input to hidden node %d is %9.4f and activation is %9.4f\n", j, in, activation[j*stride]);
#endif
	}
	
	result += theta[iOutputBias] * -1.0f;
	
#ifdef DEBUG_CALC_Q
		printf("output activation is %9.4f\n", result);
#endif
	return result;
}

// different strides for state and theta
DUAL_PREFIX float calc_Q2(float *s, unsigned a, float *theta, unsigned stride_theta, unsigned num_hidden, float *activation)
{
	// adjust theta to point to beginning of this action's weights
	theta += iActionStart(a, stride_theta, num_hidden);
	
	float result = 0.0f;
	unsigned iOutputBias = offsetToOutputBias(stride_theta, num_hidden);
	
	// loop over each hidden node
	for (int j = 0; j < num_hidden; j++) {
		// iBias is the index into theta for the bias weight for the hidden node j
		unsigned iBias = offsetToHiddenBias(j, stride_theta, num_hidden);
		
		// first calculate contribution of the bias for this hidden node
		float in = theta[iBias] * -1.0f;
		
		// next add in the contributions for the state input nodes
		for (int k = 0; k < STATE_SIZE; k++) {
			in += theta[iBias + (1+k) * stride_theta] * s[k * BLOCK_SIZE];
		}
		
		// apply sigmoid and accumulate in the result
		in = sigmoid(in); 
		if (activation) activation[j*stride_theta] = in;
		result += theta[iOutputBias + (1+j) * stride_theta] * in;

#ifdef DEBUG_CALC_Q
		printf("calc_Q for state (%9.4f, %9.4f) and action %d ... ", s[0], s[BLOCK_SIZE], a);
//		printf("input to hidden node %d is %9.4f and activation is %9.4f\n", j, in, activation[j*stride]);
#endif
	}
	
	// add in the output bias contribution
	result += theta[iOutputBias] * -1.0f;
	
#ifdef DEBUG_CALC_Q
		printf("output activation is %9.4f\n", result);
#endif
	return result;
}

DUAL_PREFIX void reset_gradient(float *W, unsigned stride, unsigned num_wgts)
{
	for (int i = 0; i < num_wgts; i++) {
		W[i * stride] = 0.0f;
	}
}

DUAL_PREFIX void accumulate_gradient(float *s, unsigned a, float *theta, unsigned stride, unsigned num_hidden, unsigned num_wgts, float *activation, float *W, float lambda, float gamma)
{
	// First, decay all the existing gradients by lambda * gamma
#ifdef DEBUG_GRADIENT_CALC
	printf("all gradients after decay:\n");
#endif
	for (int i = 0; i < num_wgts; i++) {
		W[i*stride] *= lambda * gamma;
#ifdef DEBUG_GRADIENT_CALC
		printf("   %9.6f\n", W[i*stride]);
#endif
	}


	// Next, need to add in the new gradient for the specified action.
	// adjust W & theta to point to this action's weights
	unsigned offset = iActionStart(a, stride, num_hidden);
//	printf("[accumulate_gradient] offset is %d for action %d\n", offset, a);
	theta += offset;
	W += offset;

#ifdef DEBUG_GRADIENT_CALC
	printf("updating gradients for action %d\n", a);
#endif
	
	// for gradients to output node, the gradient equals the activation of the hidden layer node (or bias) 
	// first update the gradient for bias -> output
	unsigned iOutBias = offsetToOutputBias(stride, num_hidden);
	W[iOutBias] += -1.0f;

#ifdef DEBUG_GRADIENT_CALC
	printf("[accumulate_gradient] iOutBias is %d\n", iOutBias);
	printf("output bias changed by %9.6f and is now %9.6f\n", -1.0f, W[iOutBias]);
#endif	

	// next update the gradients with respect to weights from hidden to output
	for (int j = 0; j < num_hidden; j++) {
//		printf("[accumulate_gradient] iOutBias is %d\n", iOutBias);
		W[iOutBias + (1+j)*stride] += activation[j * stride];
#ifdef DEBUG_GRADIENT_CALC
		printf("[accumulate_gradient] hidden node %d is at %d\n", j, iOutBias + (1+j)*stride);
		printf("hidden%d to output changed by %9.6f and is now %9.6f\n", j, activation[j*stride], W[iOutBias + (1+j)*stride]);
#endif
	}

	// update the gradients with respect to the weights from input to hidden
	for (int j = 0; j < num_hidden; j++) {
		// first the bias weight
		unsigned iHidBias = offsetToHiddenBias(j, stride, num_hidden);

		// gradient of output i wrt wgt from input k to hidden j equals
		// grad(in_j wrt wgt_kj) * grad(activation_j wrt in_j)     * grad(output activation wrt activation_j) = 
		//    activation_k       * activation_j * (1-activation_j) *   wgt_ji
		// The last two terms are only a function of j (and there is only one output node), so
		// calculate grad to be the last two terms
		float grad = activation[j*stride] * (1-activation[j*stride]) * theta[iOutBias + (1+j)*stride];
		
		// total gradient is the activation of the input node times grad
		// The updated value includes eligibility trace of prior gradient
		W[iHidBias] += -1.0f * grad;

#ifdef DEBUG_GRADIENT_CALC
		printf("[accumulate_gradient] iHidBias is %d\n", iHidBias);
		printf("bias to hidden%d changed by %9.6f and is now %9.6f\n", j, -1.0f*grad, W[iHidBias]);
#endif
		
		// next the states
		for (int k = 0; k < STATE_SIZE; k++) {
			W[iHidBias + (k+1)*stride] += s[k * stride] * grad;
#ifdef DEBUG_GRADIENT_CALC
			printf("[accumulate_gradient] state %d is at %d\n", k, iHidBias + (k+1)*stride);
			printf("state%d to hidden%d changed by %9.6f and is now %9.6f\n", k, j, s[k*stride]*grad, W[iHidBias + (k+1)*stride]);
#endif
		}
	}
}

DUAL_PREFIX void accumulate_gradient2(float *s, unsigned a, float *theta, unsigned stride_g, unsigned num_hidden, unsigned num_wgts, float *activation, float *W, float lambda, float gamma)
{
	// First, decay all the existing gradients by lambda * gamma
	for (int i = 0; i < num_wgts; i++) {
		W[i*stride_g] *= lambda * gamma;
	}


	// Next, need to add in the new gradient for the specified action.
	// adjust W & theta to point to this action's weights
	unsigned offset = iActionStart(a, stride_g, num_hidden);
	theta += offset;
	W += offset;

	// for gradients to output node, the gradient equals the activation of the hidden layer node (or bias) 
	// first update the gradient for bias -> output
	unsigned iOutBias = offsetToOutputBias(stride_g, num_hidden);
	W[iOutBias] += -1.0f;

	// next update the gradients with respect to weights from hidden to output
	for (int j = 0; j < num_hidden; j++) {
		W[iOutBias + (1+j)*stride_g] += activation[j * stride_g];
	}

	// update the gradients with respect to the weights from input to hidden
	for (int j = 0; j < num_hidden; j++) {
		// first the bias weight
		unsigned iHidBias = offsetToHiddenBias(j, stride_g, num_hidden);

		// gradient of output i wrt wgt from input k to hidden j equals
		// grad(in_j wrt wgt_kj) * grad(activation_j wrt in_j)     * grad(output activation wrt activation_j) = 
		//    activation_k       * activation_j * (1-activation_j) *   wgt_ji
		// The last two terms are only a function of j (and there is only one output node), so
		// calculate grad to be the last two terms
		float grad = activation[j*stride_g] * (1-activation[j*stride_g]) * theta[iOutBias + (1+j)*stride_g];
		
		// total gradient is the activation of the input node times grad
		// The updated value includes eligibility trace of prior gradient
		W[iHidBias] += -1.0f * grad;

		// next the states
		for (int k = 0; k < STATE_SIZE; k++) {
			W[iHidBias + (k+1)*stride_g] += s[k * BLOCK_SIZE] * grad;
		}
	}
}


// Update the weights in the neural net (theta's) using back-propagation of the output error
// Current activation for the hidden layer is pre-calculated in activation
DUAL_PREFIX void update_thetas(float *s, float *theta0, float *W0, float alpha, float error, unsigned stride, unsigned num_hidden, float *activation)
{	
	// Repeat for all actions
	for (int a = 0; a < NUM_ACTIONS; a++) {
		// adjust theta and W to point to start of weights/gradients for this action
		unsigned offset = iActionStart(a, stride, num_hidden);
		float *theta = theta0 + offset;
		float *W = W0 + offset;
		
		// First the bias
		// wgt_j_i += alpha * error * W_ji
		unsigned iOutBias = offsetToOutputBias(stride, num_hidden);
		theta[iOutBias] += alpha * error * W[iOutBias];
//		if (isnan(theta[iOutBias])){
//			printf("theta ISNAN !! added error of %9.6f with alpha of %9.6f\n", error, alpha);
//		}

#ifdef DEBUG_THETA_UPDATE
		printf("\nupdate_thetas for error of %9.7f\n", error);
		printf("output bias: change is alpha (%9.7f) * error (%9.7f) * gradient (%9.7f) to get new value of %9.7f\n", alpha, error, W[iOutBias], theta[iOutBias]);
#endif

		// next update each weight from hidden nodes to output node
		for (int j = 0; j < num_hidden; j++) {
			// wgt_j_i += alpha * error * W_ji
			theta[iOutBias + (1+j) * stride] += alpha * error * W[iOutBias + (1+j)*stride];
#ifdef DEBUG_THETA_UPDATE
		printf("hidden%d: change is alpha (%9.7f) * error (%9.7f) * gradient (%9.7f) to get new value of %9.7f\n", j, alpha, error, W[iOutBias + (1+j)*stride], theta[iOutBias + (1+j)*stride]);
#endif
		}
		
		// update weights from input layer to hidden layer for each node in hidden layer
		for (int j = 0; j < num_hidden; j++) {
			// first update the bias weight
			// wgt_k_j = alpha * error * W_k_j
			unsigned iHidBias = offsetToHiddenBias(j, stride, num_hidden);
			theta[iHidBias] += alpha * error * W[iHidBias];
#ifdef DEBUG_THETA_UPDATE
			printf("bias -> hidden%d: change is alpha (%9.7f) * error (%9.7f) * gradient (%9.7f) to get new value of %9.7f\n", j, alpha, error, W[iHidBias], theta[iHidBias]);
#endif
			
			// update the weights from the state nodes
			for (int k = 0; k < STATE_SIZE; k++) {
				// wgt_k_j = alpha * error * W_k_j
				theta[iHidBias + (k+1) * stride] += alpha * error * W[iHidBias + (k+1)*stride];
#ifdef DEBUG_THETA_UPDATE
			printf("state%d -> hidden%d: change is alpha (%9.7f) * error (%9.7f) * gradient (%9.7f) to get new value of %9.7f\n", k, j, alpha, error, W[iHidBias + (k+1)*stride], theta[iHidBias + (k+1)*stride]);
#endif
			}
		}
	}
}

DUAL_PREFIX void update_thetas2(float *s, float *theta0, float *W0, float alpha, float error, unsigned stride_s, unsigned stride_g, unsigned num_hidden, float *activation)
{	
	// Repeat for all actions
	for (int a = 0; a < NUM_ACTIONS; a++) {
		// adjust theta and W to point to start of weights/gradients for this action
		unsigned offset = iActionStart(a, stride_g, num_hidden);
		float *theta = theta0 + offset;
		float *W = W0 + offset;
		
		// First the bias
		unsigned iOutBias = offsetToOutputBias(stride_g, num_hidden);
		theta[iOutBias] += alpha * error * W[iOutBias];

		// next update each weight from hidden nodes to output node
		for (int j = 0; j < num_hidden; j++) {
			// wgt_j_i += alpha * error * W_ji
			theta[iOutBias + (1+j) * stride_g] += alpha * error * W[iOutBias + (1+j)*stride_g];
		}
		
		// update weights from input layer to hidden layer for each node in hidden layer
		for (int j = 0; j < num_hidden; j++) {
			// first update the bias weight
			// wgt_k_j = alpha * error * W_k_j
			unsigned iHidBias = offsetToHiddenBias(j, stride_g, num_hidden);
			theta[iHidBias] += alpha * error * W[iHidBias];
			
			// update the weights from the state nodes
			for (int k = 0; k < STATE_SIZE; k++) {
				// wgt_k_j = alpha * error * W_k_j
				theta[iHidBias + (k+1) * stride_g] += alpha * error * W[iHidBias + (k+1)*stride_g];
			}
		}
	}
}



// Calculate the Q value for each action from the given state, returning the best Q value
// and storing the action in *pAction
DUAL_PREFIX float best_action(float *s, unsigned *pAction, float *theta, unsigned stride, unsigned num_hidden, float *activation)
{
	// calculate Q value for each action
	unsigned best_action = 0;
	float bestQ = calc_Q(s, 0, theta, stride, num_hidden, activation);
	for (int k = 1; k < NUM_ACTIONS; k++) {
		float tempQ = calc_Q(s, k, theta, stride, num_hidden, activation);
		if (tempQ > bestQ) {
			bestQ = tempQ;
			best_action = k;
		}
	}
	*pAction = best_action;
	return bestQ;
}

DUAL_PREFIX float best_action2(float *s, unsigned *pAction, float *theta, unsigned stride_g, unsigned num_hidden, float *activation)
{
	// calculate Q value for each action
	unsigned best_action = 0;
	float bestQ = calc_Q2(s, 0, theta, stride_g, num_hidden, activation);
	for (int k = 1; k < NUM_ACTIONS; k++) {
		float tempQ = calc_Q2(s, k, theta, stride_g, num_hidden, activation);
		if (tempQ > bestQ) {
			bestQ = tempQ;
			best_action = k;
		}
	}
	*pAction = best_action;
	return bestQ;
}

// choose action from current state, return the Q value for the chosen action
// and store the action in *pAction
DUAL_PREFIX float choose_action(float *s, unsigned *pAction, float *theta, float epsilon, unsigned stride, unsigned num_hidden, float *activation, unsigned *seeds)
{
	if (epsilon > 0.0f && RandUniform(seeds, stride) < epsilon){
		// choose random action
		float r = RandUniform(seeds, stride);
		*pAction = r * NUM_ACTIONS;
		return calc_Q(s, *pAction, theta, stride, num_hidden, activation);
	}else{
		// choose the best action
		return best_action(s, pAction, theta, stride, num_hidden, activation);
	}
}

DUAL_PREFIX float choose_action2(float *s, unsigned *pAction, float *theta, float epsilon, unsigned stride_g, unsigned num_hidden, float *activation, unsigned *seeds)
{
	if (epsilon > 0.0f && RandUniform(seeds, BLOCK_SIZE) < epsilon){
		// choose random action
		float r = RandUniform(seeds, BLOCK_SIZE);
		*pAction = r * NUM_ACTIONS;
		return calc_Q2(s, *pAction, theta, stride_g, num_hidden, activation);
	}else{
		// choose the best action
		return best_action2(s, pAction, theta, stride_g, num_hidden, activation);
	}
}

DUAL_PREFIX unsigned terminal_state(float *s)
{
	return s[0] >= MAX_X;
}

// take an action from the current state, s, returning the reward and saving new state in s_prime
// Note, s & s_prime may be the same location.
DUAL_PREFIX float take_action(float *s, unsigned a, float *s_prime, unsigned stride, float *accel)
{
	// Forumlation of mountain car problem is from Sutton & Barto, 
	// "Reinforcement Learning, An Introduction"
	
#ifdef DEBUG_CPU
	printf("take_action %s from state (%9.4f, %9.4f)\n", string_for_action(a), s[0], s[stride]);
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
	printf("new state is (%9.6f, %9.6f) and reward is %9.6f\n", s_prime[0], s_prime[stride], reward);
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

//DUAL_PREFIX void randomize_state2(float *s, unsigned *seeds, unsigned stride_s, unsigned stride_g)
//{
//	s[0] = rand_in_range(seeds, stride_g, MIN_X, MAX_X);
//	s[stride_s] = rand_in_range(seeds, stride_g, MIN_VEL, MAX_VEL);
//}

__device__ void randomize_stateGPU(unsigned ag)
{
	dc_ag.s[ag] = rand_in_range(dc_ag.seeds + ag, dc_p.stride, MIN_X, MAX_X);
	dc_ag.s[ag + dc_p.stride] = rand_in_range(dc_ag.seeds + ag, dc_p.stride, MIN_VEL, MAX_VEL);
}

void randomize_all_states(AGENT_DATA *ag)
{
	// randomize state for all agents, deterine first action and set activation values for hidden 
	for (int agent = 0; agent < _p.agents; agent++) {
		randomize_state(ag->s + agent, ag->seeds + agent, _p.agents);
		reset_gradient(ag->W + agent, _p.agents, _p.num_wgts);
//		printf("randomize_state, state is now (%9.6f, %9.6f)\n", ag->s[agent], ag->s[agent + _p.agents]);
		choose_action(ag->s + agent, ag->action + agent, ag->theta + agent, _p.epsilon, _p.agents, _p.hidden_nodes, ag->activation + agent, ag->seeds + agent);
		// force activation values to be recalculated for the chosen action
//		printf("chosen action will be %s\n", string_for_action(ag->action[agent]));
		calc_Q(ag->s + agent, ag->action[agent], ag->theta + agent, _p.agents, _p.hidden_nodes, ag->activation + agent);
		// update_trace(...
	}
}


#pragma mark -
#pragma mark CPU
void set_params(PARAMS p){
	_p = p;

	// fill in some calculated values in the parameters
	_p.iActionStart[0] = iActionStart(0, p.agents, p.hidden_nodes);
	_p.iActionStart[1] = iActionStart(1, p.agents, p.hidden_nodes);
	_p.iActionStart[2] = iActionStart(2, p.agents, p.hidden_nodes);

	printf("iActionStart values %d, %d, %d\n", _p.iActionStart[0], _p.iActionStart[1], _p.iActionStart[2]);
	
	_p.offsetToOutputBias = offsetToOutputBias(p.agents, p.hidden_nodes);
	printf("_p.agents = %d, _p.hidden_nodes = %d\n", _p.agents, _p.hidden_nodes);
	
}

// dump agent data to stdout
// uses parameter values in _p
// (hard-coded to 2 dimensional state)
void dump_agent(AGENT_DATA *ag, unsigned agent)
{
	printf("[agent %d]: ", agent);
	printf("   seeds = %u, %u, %u, %u\n", ag->seeds[agent], ag->seeds[agent + _p.agents], 
									   ag->seeds[agent + 2*_p.agents], ag->seeds[agent + 3*_p.agents]);
	printf("          FROM          TO       THETA       W  \n");
	unsigned i = agent;
	for (int a = 0; a < NUM_ACTIONS; a++) {
		for (int h = 0; h < _p.hidden_nodes; h++) {
			printf("[%6s]    bias --> hidden%2d %9.6f %9.6f\n", string_for_action(a), h, ag->theta[i], ag->W[i]); i += _p.agents;
			printf("              x  --> hidden%2d %9.6f %9.6f\n", h, ag->theta[i], ag->W[i]); i += _p.agents;
			printf("              x' --> hidden%2d %9.6f %9.6f\n", h, ag->theta[i], ag->W[i]); i += _p.agents;
		}
		printf(    "            bias --> output   %9.6f %9.6f\n", ag->theta[i], ag->W[i]); i += _p.agents;
		for (int h = 0; h < _p.hidden_nodes; h++) {
			printf("        hidden%2d --> output   %9.6f %9.6f\n", h, ag->theta[i], ag->W[i]); i += _p.agents;
		}
	}
	printf("fitness = %5.0f\n", ag->fitness[agent]);
	printf("\nCurrent State: x = %9.6f  x' = %9.6f, stored action is %s\n", ag->s[agent], ag->s[agent + _p.agents], string_for_action(ag->action[agent]));

	printf("        HIDDEN NODE    ACTIVATION\n");
	for (int j = 0; j < _p.hidden_nodes; j++) {
		printf("[%6s]   %3d      %9.6f\n", string_for_action(ag->action[agent]), j, ag->activation[agent + j * _p.agents]);
	}
	printf("\n");
}

void dump_agent_pointers(const char *str, AGENT_DATA *ag)
{
	printf("\n===================================================\n%s\n", str);
	printf("---------------------------------------------------\n", str);
	printf("     seeds: %p\n", ag->seeds);
	printf("     theta: %p\n", ag->theta);
	printf("         W: %p\n", ag->W);
	printf("     state: %p\n", ag->s);
	printf("activation: %p\n", ag->activation);
	printf("    action: %p\n", ag->action);
	printf("   fitness: %p\n", ag->fitness);
	printf("====================================================\n\n", str);
}

// print message and dump all agent data
void dump_agents(const char *str, AGENT_DATA *ag)
{
	printf("\n===================================================\n%s\n", str);
	printf("---------------------------------------------------\n", str);
	for (int agent = 0; agent < _p.agents; agent++) {
		dump_agent(ag, agent);
	}
	printf("====================================================\n\n", str);
}

void dump_one_agent(const char *str, AGENT_DATA *ag)
{
	printf("%s\n", str);
	dump_agent(ag, 0);
}


RESULTS *initialize_results()
{
	RESULTS *r = (RESULTS *)malloc(sizeof(RESULTS));
	r->avg_fitness = (float *)malloc(_p.num_tests * sizeof(float));
	r->best_fitness = (float *)malloc(_p.num_tests * sizeof(float));
	r->best_agent = (unsigned *)malloc(_p.num_tests * sizeof(unsigned));
	return r;
}

void free_results(RESULTS *r)
{
	if (r){
		if (r->avg_fitness){ free(r->avg_fitness); r->avg_fitness = NULL;}
		if (r->best_fitness){ free(r->best_fitness); r->best_fitness = NULL;}
		if (r->best_agent){ free(r->best_agent); r->best_agent = NULL;}
		free(r);
	}
}

void display_results(const char *str, RESULTS *r)
{
	printf("%s \n", str);
	printf("    TEST  Avg Steps\n");
	for (int i = 0; i < _p.num_tests; i++) {
		printf("   [%10d]%8.0f, %8.0f, %8d\n", i*_p.test_interval, r->avg_fitness[i], r->best_fitness[i], r->best_agent[i]);
	}
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

// create wgts set initially to random values between theta_min and theta_max
float *create_theta(unsigned num_agents, unsigned num_wgts, float theta_min, float theta_max)
{
#ifdef VERBOSE
	printf("create_theta for %d agents and %d weights in range %9.7f to %9.7f\n", num_agents, num_wgts, theta_min, theta_max);
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

unsigned *create_actions(unsigned num_agents, unsigned num_actions)
{
#ifdef VERBOSE
	printf("create_actions for %d agents\n", num_agents);
#endif
	unsigned *actions = (unsigned *)malloc(num_agents * num_actions * sizeof(unsigned));
	for (int i = 0; i < num_agents * num_actions; i++) actions[i] = num_actions; // not valid value
	return actions;
}

float *create_fitness(unsigned num_agents)
{
	float *fitness = (float *)malloc(num_agents * sizeof(float));
	for (int i = 0; i < num_agents; i++) fitness[i] = MAX_FITNESS;
	return fitness;
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

// initialize agents on CPU, including the initial randomization of state and choice of first action
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
//	ag->Q = create_Q(_p.agents, _p.num_actions);
	ag->action = create_actions(_p.agents, _p.num_actions);
	ag->activation = create_activation(_p.agents, _p.hidden_nodes);
	ag->fitness = create_fitness(_p.agents);
	
	randomize_all_states(ag);
	
	return ag;
}

void free_agentsCPU(AGENT_DATA *ag)
{
#ifdef VERBOSE
	printf("freeing agents on CPU...\n");
#endif
	if (ag) {
		if (ag->seeds){ free(ag->seeds); ag->seeds = NULL;}
		if (ag->theta){ free(ag->theta); ag->theta = NULL;}
		if (ag->W){ free(ag->W); ag->W = NULL;}
		if (ag->s){ free(ag->s); ag->s = NULL;}
		if (ag->action){ free(ag->action); ag->action = NULL;}
		if (ag->activation){ free(ag->activation); ag->activation = NULL;}
		if (ag->fitness){ free(ag->fitness); ag->fitness = NULL;}
		free(ag);
	}
}

/*
	On entry, the agent data has the current state and chosen action based on current weights.
 */
void learning_session(AGENT_DATA *ag)
{
	// for each time step
	for (int t = 0; t < _p.chunk_interval; t++) {

#ifdef VERBOSE
		printf("\n*****************************************\n");
		printf(  "************ TIME STEP %d ****************\n", t);
		printf(  "*****************************************\n");
#endif

		// for each agent
		for (int agent = 0; agent < _p.agents; agent++) {
#ifdef DEBUG_CPU
			printf("[[ AGENT %d ]]\n", agent);
#endif
			// Calculate Q_curr based on current state and action
			// Activation values will be stored for use in updating the gradient
			float Q_curr = calc_Q(ag->s + agent, ag->action[agent], ag->theta + agent, _p.agents, _p.hidden_nodes, ag->activation + agent);
#ifdef DEBUG_CPU
			printf("Q_curr is %9.6f based on state (%9.6f, %9.6f) and action %s\n", Q_curr, ag->s[agent], ag->s[agent + _p.agents], string_for_action(ag->action[agent]));
#endif
			
			//accumulate_gradient uses current activations and weights to update the gradient array, W 
			accumulate_gradient(ag->s + agent, ag->action[agent], ag->theta + agent, _p.agents, _p.hidden_nodes, _p.num_wgts, ag->activation + agent, ag->W + agent, _p.lambda, _p.gamma);

//#ifdef DUMP_AGENT_UPDATES
//		dump_agents("after accumulate_gradient", ag);
//#endif

			// take_action will calculate the new state based on the current state and current action,
			// storing the new state in the agent, returning the reward
			float reward = take_action(ag->s + agent, ag->action[agent], ag->s + agent, _p.agents, accel);
#ifdef DUMP_STATES
			printf("[AGENT%3d] x = %9.6f  x' = %9.6f  after action = %s\n", agent, ag->s[agent], ag->s[agent + _p.agents], string_for_action(ag->action[agent]));
#endif

			unsigned success = terminal_state(ag->s + agent);
			if (success){
//				printf("success for ageent %d at time step %d\n", agent, t);
				randomize_state(ag->s + agent, ag->seeds + agent, _p.agents);
			}

			// choose the next action, storing it in the agent and returning the Q_next value
			float Q_next = choose_action(ag->s + agent, ag->action + agent, ag->theta + agent, _p.epsilon, _p.agents, _p.hidden_nodes, ag->activation + agent, ag->seeds + agent);
#ifdef DEBUG_CPU
			printf("Q_next is %12.6f based on state (%9.6f, %9.6f) and action %s\n", Q_next, ag->s[agent], ag->s[agent + _p.agents], string_for_action(ag->action[agent]));
#endif
			float error = reward + _p.gamma*Q_next - Q_curr;
//			printf("reward + _p.gamma*Q_next = %9.6f, (Q_next is %9.6f), Q_curr = %9.6f, so error is %9.6f\n", reward + _p.gamma*Q_next, Q_next, Q_curr, error);
#ifdef DEBUG_CPU
			printf("error is %12.6f\n", error);
#endif
			update_thetas(ag->s + agent, ag->theta + agent, ag->W + agent, _p.alpha, error, _p.agents, _p.hidden_nodes, ag->activation + agent);

//#ifdef DUMP_AGENT_UPDATES
//		dump_agents("after update_thetas", ag);
//#endif
			if (success) reset_gradient(ag->W + agent, _p.agents, _p.num_wgts);
		}	
		
#ifdef DUMP_AGENT_UPDATES
		printf("***** end of time step %d *****\n", t);
		dump_agents("after update_thetas", ag);
#endif
			
//			update_stored_Q(ag->Q + agent, ag->s + agent, ag->theta + agent, _p.agents, _p.state_size, _p.num_actions, _p.hidden_nodes, ag->activation + agent);
//			update_trace(...
	}
}

// copy theta valuels from agent iFrom and over-write agent iTo
void copy_theta(AGENT_DATA *ag, unsigned iFrom, unsigned iTo, unsigned num_wgts, unsigned stride)
{
	for (int i = 0; i < num_wgts; i++) {
		ag->theta[iTo + i * stride] = ag->theta[iFrom + i*stride];
	}
}

// share is where the best agents will be selected and duplicated
void share(AGENT_DATA *ag, float share_best_pct, unsigned agent_group_size, unsigned num_agents, unsigned num_wgts)
{
//	printf("share...\n");
	for (int group = 0; group < num_agents / agent_group_size; group++) {
		unsigned iGroup = group * agent_group_size;
		// determine the best agent in this group
		unsigned iBest = 0;
		float best_fitness = ag->fitness[iGroup];
		for (int a = 1; a < agent_group_size; a++) {
			if (ag->fitness[iGroup + a] < best_fitness) {
				best_fitness = ag->fitness[iGroup + a];
				iBest = a;
			}
		}

//		printf("agent %d is the best in group %d\n", iGroup + iBest, group);
		
		// now copy the best agents to the others with probability share_best_pct
		for (int a = 0; a < agent_group_size; a++) {
			if (a == iBest) continue;
			float r = RandUniform(ag->seeds + iGroup + a, num_agents);
			if (r < share_best_pct) {
//				printf("copy weights from agent %d to agent %d\n", iGroup + iBest, iGroup + a);
				copy_theta(ag, iBest, iGroup + a, num_wgts, num_agents);
			}
		}
	}
}

// test the agents and store the results in the iTest entry in the RESULTS arrays
void run_test(AGENT_DATA *ag, RESULTS *r, unsigned iTest)
{
	float total_steps = 0.0f;
	float best_fitness = MAX_FITNESS;
	unsigned best_agent = 999999;
	
	float save_s[STATE_SIZE];
	unsigned save_action;			//**TODO** may not need to be saved
	unsigned save_seeds[4];
	
	static float *junk_activation = NULL;
	if(!junk_activation) junk_activation = (float *)malloc(_p.hidden_nodes * sizeof(float));
	
	// test all agents and average the result
	for (int agent = 0; agent < _p.agents; agent++) {
#ifdef TRACE_TEST
		printf("Testing agent %d...\n", agent);
#endif
		// save agent state prior to testing
		save_s[0] = ag->s[agent];
		save_s[1] = ag->s[agent + _p.agents];
		save_action = ag->action[agent];
		save_seeds[0] = ag->seeds[agent];
		save_seeds[1] = ag->seeds[agent + _p.agents];
		save_seeds[2] = ag->seeds[agent + 2*_p.agents];
		save_seeds[3] = ag->seeds[agent + 3*_p.agents];
		
		float agent_steps = 0.0f;
		
		for (int rep = 0; rep < _p.test_reps; rep++) {
		
			ag->seeds[agent] = save_seeds[0] + rep;
			ag->seeds[agent + _p.agents] = save_seeds[1] + rep;
			ag->seeds[agent + 2*_p.agents] = save_seeds[2] + rep;
			ag->seeds[agent + 3*_p.agents] = save_seeds[3] + rep;

			randomize_state(ag->s + agent, ag->seeds + agent, _p.agents);
			int t;
			unsigned action;
			for (t = 0; t < _p.test_max; t++) {
				best_action(ag->s + agent, &action, ag->theta + agent, _p.agents, _p.hidden_nodes, junk_activation);			
#ifdef TRACE_TEST
				printf("[test%4d] state = (%9.6f, %9.6f) action will be %s\n", t, ag->s[agent], ag->s[agent + _p.agents], string_for_action(action));
#endif
				take_action(ag->s + agent, action, ag->s + agent, _p.agents, accel);
				if (terminal_state(ag->s + agent)) {
#ifdef TRACE_TEST
					printf("Done at step %d!!!\n", t);
#endif
					break;
				}
			}
#ifdef TRACE_TEST
			if (t == _p.test_reps) printf("failure\n");
#endif
			agent_steps += t;
		}

		ag->fitness[agent] = agent_steps / _p.test_reps;
		if (ag->fitness[agent] < best_fitness){
			best_fitness = ag->fitness[agent];
			best_agent = agent;
		}
		total_steps += agent_steps;

		//restore state and action
		ag->s[agent] = save_s[0];
		ag->s[agent + _p.agents] = save_s[1];
		ag->action[agent] = save_action;
		ag->seeds[agent] = save_seeds[0];
		ag->seeds[agent + _p.agents] = save_seeds[1];
		ag->seeds[agent + 2*_p.agents] = save_seeds[2];
		ag->seeds[agent + 3*_p.agents] = save_seeds[3];
	}
	
#ifdef DUMP_TESTED_AGENTS
	printf("Testing %d\n", iTest);
	dump_agents("after testing", ag);
#endif	

	r->avg_fitness[iTest] = total_steps / float(_p.agents) / float(_p.test_reps);
	r->best_fitness[iTest] = best_fitness;
	r->best_agent[iTest] = best_agent;
}


void run_CPU(AGENT_DATA *ag, RESULTS *r)
{
#ifdef VERBOSE
	printf("\n==============================================\nrunning on CPU...\n");
#endif

//	dump_agents("run_CPU entry", ag);
	
	unsigned timer;
	CREATE_TIMER(&timer);
	START_TIMER(timer);

	timing_feedback_header(_p.num_chunks);

	for (int i = 0; i < _p.num_chunks; i++) {

		timing_feedback_dot(i);
		
		if ((i > 0) && 0 == (i % _p.chunks_per_restart)){
//			printf("randomize all states...\n");
			randomize_all_states(ag);
#ifdef DUMP_AGENTS_AFTER_RESTART
			dump_agents("after restart", ag);
#endif
		}
		
		if (i == 0) {
#ifdef DUMP_INITIAL_AGENTS
			dump_agents("Initial agents on CPU, prior to learning session", ag);
#endif
//			run_test(ag, r, i);
		}
		
		learning_session(ag);
//		dump_agents("after learning session", ag);
		
		if (0 == ((i+1) % _p.chunks_per_test)) run_test(ag, r, (i+1)/_p.chunks_per_test);
		
//		dump_agents("after testing", ag);

//		if ((_p.agent_group_size > 1) && 0 == ((i+1) % _p.chunks_per_share)) {
//			share(ag, _p.share_best_pct, _p.agent_group_size, _p.agents, _p.num_wgts);
//		}
	}
	printf("\n");
	STOP_TIMER(timer, "run on CPU");

//#ifdef DUMP_FINAL_AGENTS
//	dump_agents("Final agents on CPU", ag);
//#endif

}

#pragma mark -
#pragma mark GPU

// copy agents from device back to host
AGENT_DATA *copy_GPU_agents(AGENT_DATA *agGPU)
{
//	printf("copy_GPU_agents\n");
	AGENT_DATA *agCopy = (AGENT_DATA *)malloc(sizeof(AGENT_DATA));
//	dump_agent_pointers("agGPU", agGPU);
//	printf("  %d seeds from %p\n", _p.agents * 4, agGPU->seeds);
	agCopy->seeds = host_copyui(agGPU->seeds, _p.agents * 4);
	agCopy->theta = host_copyf(agGPU->theta, _p.agents * _p.num_wgts);
	agCopy->W = host_copyf(agGPU->W, _p.agents * _p.num_wgts);
	agCopy->s = host_copyf(agGPU->s, _p.agents * _p.state_size);
	agCopy->activation = host_copyf(agGPU->activation, _p.agents * _p.hidden_nodes);
	agCopy->action = host_copyui(agGPU->action, _p.agents);
	agCopy->fitness = host_copyf(agGPU->fitness, _p.agents);
	return agCopy;
}

// dump the agents from the GPU by first copying to CPU and then dumping the CPU copy
void dump_agentsGPU(const char *str, AGENT_DATA *agGPU)
{
	AGENT_DATA *agCopy = copy_GPU_agents(agGPU);
	dump_agents(str, agCopy);
	free(agCopy);
}

void dump_one_agentGPU(const char *str, AGENT_DATA *agGPU, unsigned ag)
{
	AGENT_DATA *agCopy = copy_GPU_agents(agGPU);
	printf("%s\n", str);
	dump_agent(agCopy, ag);
	free(agCopy);
}

// Copy the provided CPU agent data to the GPU, storing device pointers in a new AGENT_DATA structure
// Also copy the AGENT_DATA and PARAMS structures to constant memory on the device
AGENT_DATA *initialize_agentsGPU(AGENT_DATA *agCPU)
{
#ifdef VERBOSE
	printf("\n==============================================\nrunning on GPU...\n");
#endif
#ifdef VERBOSE
	printf("initializing agents on GPU...\n");
#endif
	AGENT_DATA *ag = (AGENT_DATA *)malloc(sizeof(AGENT_DATA));
	ag->seeds = device_copyui(agCPU->seeds, _p.agents * 4);
	ag->theta = device_copyf(agCPU->theta, _p.agents * _p.num_wgts);
	ag->W = device_copyf(agCPU->W, _p.agents * _p.num_wgts);
	ag->s = device_copyf(agCPU->s, _p.agents * _p.state_size);
	ag->activation = device_copyf(agCPU->activation, _p.agents * _p.hidden_nodes);
	ag->action = device_copyui(agCPU->action, _p.agents);
	ag->fitness = device_copyf(agCPU->fitness, _p.agents);
	
	cudaMemcpyToSymbol("dc_p", &_p, sizeof(PARAMS));
	cudaMemcpyToSymbol("dc_ag", ag, sizeof(AGENT_DATA));
	cudaMemcpyToSymbol("dc_accel", accel, 3 * sizeof(float));
	
//	dump_agent_pointers("agent copied to GPU", ag);
	
	return ag;
}

// Free the deivce memory pointed to by elements of AGENT_DATA ag, then free ag
void free_agentsGPU(AGENT_DATA *ag)
{
#ifdef VERBOSE
	printf("freeing agents on GPU...\n");
#endif
	if (ag) {
		if (ag->seeds){ cudaFree(ag->seeds); ag->seeds = NULL;}
		if (ag->theta){ cudaFree(ag->theta); ag->theta = NULL;}
		if (ag->W){ cudaFree(ag->W); ag->W = NULL;}
		if (ag->s){ cudaFree(ag->s); ag->s = NULL;}
		if (ag->activation){ cudaFree(ag->activation); ag->activation = NULL;}
		if (ag->action){ cudaFree(ag->action); ag->action = NULL;}
		if (ag->fitness){ cudaFree(ag->fitness); ag->fitness = NULL;}
		free(ag);
	}
}

__global__ void kernel_randomize_all_states()
{
}

__global__ void reset_gradient_kernel()
{
	unsigned iGlobal = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	if (iGlobal < dc_p.agents * dc_p.num_wgts) dc_ag.W[iGlobal] = 0.0f;
}


/*
	threads per block = number of test reps
	number of blocks = total number of agents (agents_per_group * trials)
 */	
__global__ void test_kernel2(float *results)
{
	unsigned idx = threadIdx.x;
//	if (idx >= dc_p.test_reps) return;
	
	unsigned iGlobal = blockIdx.x + blockIdx.y * gridDim.x;
	
	__shared__ unsigned s_seeds[4*BLOCK_SIZE];
	__shared__ float s_s[2*BLOCK_SIZE];
	__shared__ float s_fitness[BLOCK_SIZE];
	
	// first copy the agent's seeds to shared, adding in the idx value so each test starts
	// with different seeds
	s_seeds[idx] = dc_ag.seeds[iGlobal] + idx;
	s_seeds[idx + BLOCK_SIZE] = dc_ag.seeds[iGlobal + dc_p.agents] + idx;
	s_seeds[idx + 2*BLOCK_SIZE] = dc_ag.seeds[iGlobal + 2*dc_p.agents] + idx;
	s_seeds[idx + 3*BLOCK_SIZE] = dc_ag.seeds[iGlobal + 3*dc_p.agents] + idx;
	
	// randomize the state
	randomize_state(s_s + idx, s_seeds + idx, BLOCK_SIZE);
	
	unsigned action;
	int t = 0;
	// only do the test for the actual number of test_reps
	if (idx < dc_p.test_reps) {
		for (t = 0; t < dc_p.test_max; t++) {
			best_action2(s_s + idx, &action, dc_ag.theta + iGlobal, dc_p.agents, dc_p.hidden_nodes, NULL);
			take_action(s_s + idx, action, s_s+idx, BLOCK_SIZE, dc_accel);
			if (terminal_state(s_s + idx)) break;
		}
	}
	s_fitness[idx] = (float)t;
	__syncthreads();
	
	// now calculate the average fitness by doing a reduction
	unsigned half = BLOCK_SIZE / 2;
	while (half > 0) {
		if (idx < half && idx + half < dc_p.test_reps) {
			s_fitness[idx] += s_fitness[idx + half];
		}
		half /= 2;
		__syncthreads();
	}
	
	// copy the fitness value to global memory for this agent, and store in results
	if (idx == 0) {
		dc_ag.fitness[iGlobal] = s_fitness[0] / (float)dc_p.test_reps;
		results[iGlobal] = dc_ag.fitness[iGlobal];
	}
	
}

__global__ void test_kernel(float *results)
{
	unsigned iGlobal = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	unsigned idx = threadIdx.x;
	if (iGlobal >= dc_p.agents) return;
	
	__shared__ float s_s[2 * BLOCK_SIZE];		// assumes state size is 2
	__shared__ unsigned s_action[BLOCK_SIZE];
	__shared__ unsigned s_seeds[4*BLOCK_SIZE];
	 
	// copy state from global to shared memory to preserve it
	s_s[idx] = dc_ag.s[iGlobal];
	s_s[idx + BLOCK_SIZE] = dc_ag.s[iGlobal + dc_p.stride];
	s_action[idx] = dc_ag.action[iGlobal];
	s_seeds[idx] = dc_ag.seeds[iGlobal];
	s_seeds[idx + BLOCK_SIZE] = dc_ag.seeds[iGlobal + dc_p.stride];
	s_seeds[idx + 2*BLOCK_SIZE] = dc_ag.seeds[iGlobal + 2*dc_p.stride];
	s_seeds[idx + 3*BLOCK_SIZE] = dc_ag.seeds[iGlobal + 3*dc_p.stride];
	 
	float steps = 0.0f;
	for (int rep = 0; rep < dc_p.test_reps; rep++) {
		// set the global seeds to original value + the rep number
		// to be consistent with test_kernel2
		dc_ag.seeds[iGlobal] = s_seeds[idx] + rep;
		dc_ag.seeds[iGlobal + dc_p.stride] = s_seeds[idx + BLOCK_SIZE] + rep;
		dc_ag.seeds[iGlobal + 2*dc_p.stride] = s_seeds[idx + 2*BLOCK_SIZE] + rep;
		dc_ag.seeds[iGlobal + 3*dc_p.stride] = s_seeds[idx + 3*BLOCK_SIZE] + rep;
		
		randomize_state(dc_ag.s + iGlobal, dc_ag.seeds + iGlobal, dc_p.stride);
		int t;
		unsigned action;
		for (t = 0; t < dc_p.test_max; t++) {
			best_action(dc_ag.s + iGlobal, &action, dc_ag.theta + iGlobal, dc_p.stride, dc_p.hidden_nodes, dc_ag.activation + iGlobal);
			take_action(dc_ag.s + iGlobal, action, dc_ag.s + iGlobal, dc_p.stride, dc_accel);
			if (terminal_state(dc_ag.s + iGlobal)) break;
		}
		steps += (float)t;
	}
	dc_ag.fitness[iGlobal] = steps / dc_p.test_reps;
	
	// restore the state from shared memory
	dc_ag.s[iGlobal] = s_s[idx];
	dc_ag.s[iGlobal + dc_p.stride] = s_s[idx + BLOCK_SIZE];
	dc_ag.action[iGlobal] = s_action[idx];
	dc_ag.seeds[iGlobal] = s_seeds[idx];
	dc_ag.seeds[iGlobal + dc_p.stride] = s_seeds[idx + BLOCK_SIZE];
	dc_ag.seeds[iGlobal + 2*dc_p.stride] = s_seeds[idx + 2*BLOCK_SIZE];
	dc_ag.seeds[iGlobal + 3*dc_p.stride] = s_seeds[idx + 3*BLOCK_SIZE];
	
	// save the result in the result array
	results[iGlobal] = dc_ag.fitness[iGlobal];
}

__global__ void learn_kernel(unsigned steps, unsigned isRestart)
{
	unsigned iGlobal = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
	unsigned idx = threadIdx.x;

	if (iGlobal >= dc_p.agents) return;
	
	__shared__ float s_s[2*BLOCK_SIZE];
	__shared__ unsigned s_action[BLOCK_SIZE];
	__shared__ unsigned s_seeds[4*BLOCK_SIZE];
	
	// copy state, action, and seeds to shared memory
	s_s[idx] = dc_ag.s[iGlobal];
	s_s[idx + BLOCK_SIZE] = dc_ag.s[iGlobal + dc_p.agents];
	s_action[idx] = dc_ag.action[iGlobal];
	s_seeds[idx] = dc_ag.seeds[iGlobal];
	s_seeds[idx + BLOCK_SIZE] = dc_ag.seeds[iGlobal + dc_p.agents];
	s_seeds[idx + 2*BLOCK_SIZE] = dc_ag.seeds[iGlobal + 2*dc_p.agents];
	s_seeds[idx + 3*BLOCK_SIZE] = dc_ag.seeds[iGlobal + 3*dc_p.agents];
	
	if (isRestart) {
//		randomize_stateGPU(iGlobal);
//		randomize_state(dc_ag.s + iGlobal, dc_ag.seeds + iGlobal, dc_p.stride);
		randomize_state(s_s + idx, s_seeds + idx, BLOCK_SIZE);
		// gradients have already been reset
		choose_action2(s_s + idx, s_action + idx, dc_ag.theta + iGlobal, dc_p.epsilon, dc_p.agents, dc_p.hidden_nodes, dc_ag.activation + iGlobal, s_seeds + idx);
		calc_Q2(s_s + idx, s_action[idx], dc_ag.theta + iGlobal, dc_p.agents, dc_p.hidden_nodes, dc_ag.activation + iGlobal);
	}
	
	for (int t = 0; t < steps; t++) {
		float Q_curr = calc_Q2(s_s + idx, s_action[idx], dc_ag.theta + iGlobal, dc_p.agents, dc_p.hidden_nodes, dc_ag.activation + iGlobal);
		accumulate_gradient2(s_s + idx, s_action[idx], dc_ag.theta + iGlobal, dc_p.agents, dc_p.hidden_nodes, dc_p.num_wgts, dc_ag.activation + iGlobal, dc_ag.W + iGlobal, dc_p.lambda, dc_p.gamma);
		float reward = take_action(s_s + idx, s_action[idx], s_s + idx, BLOCK_SIZE, dc_accel);
		unsigned success = terminal_state(s_s + idx);
		
		if (success) {
			randomize_state(s_s + idx, s_seeds + idx, BLOCK_SIZE);
		}
		float Q_next = choose_action2(s_s + idx, s_action + idx, dc_ag.theta + iGlobal, dc_p.epsilon, dc_p.agents, dc_p.hidden_nodes, dc_ag.activation + iGlobal, s_seeds + idx);
		float error = reward + dc_p.gamma * Q_next - Q_curr;
		update_thetas(NULL, dc_ag.theta + iGlobal, dc_ag.W + iGlobal, dc_p.alpha, error, dc_p.agents, dc_p.hidden_nodes, dc_ag.activation + iGlobal);
		if (success) reset_gradient(dc_ag.W + iGlobal, dc_p.agents, dc_p.num_wgts);
	}
	
	// copy state, action and seeds back to global memory
	dc_ag.s[iGlobal] = s_s[idx];
	dc_ag.s[iGlobal + dc_p.agents] = s_s[idx + BLOCK_SIZE];
	dc_ag.action[iGlobal] = s_action[idx];
	dc_ag.seeds[iGlobal] = s_seeds[idx];
	dc_ag.seeds[iGlobal + dc_p.agents] = s_seeds[idx + BLOCK_SIZE];
	dc_ag.seeds[iGlobal + 2*dc_p.agents] = s_seeds[idx + 2*BLOCK_SIZE];
	dc_ag.seeds[iGlobal + 3*dc_p.agents] = s_seeds[idx + 3*BLOCK_SIZE];
}

// run on the GPU, storing results in the RESULTS array provided
void run_GPU(AGENT_DATA *agGPU, RESULTS *rGPU)
{
	// on entry, device pointers are stored in dc_ag for agent data, and
	// parameters are stored in dc_p

//	dump_agentsGPU("run_GPU entry", agGPU);
	
	// allocate memory on device to hold results
	float *d_results = device_allocf(_p.agents * _p.num_tests);

	// calculate block and grid sizes for kernels	
	// basic arrangement has one thread for each agent in each trial
	dim3 blockDim(BLOCK_SIZE);
	dim3 gridDim(1 + (_p.agents-1) / BLOCK_SIZE);
	if (gridDim.x > 65535) {
		gridDim.y = 1 + (gridDim.x-1) / 65535;
		gridDim.x = 1 + (gridDim.x-1) / gridDim.y;
	}
	
	
	// The grid y dimension is multiplied by the number of test reps
	if (_p.test_reps > BLOCK_SIZE) printf("***** too many test_reps (%d reps greater than blocksize=%d) *****\n", _p.test_reps, BLOCK_SIZE);
	dim3 testBlockDim(_p.test_reps);
	dim3 testGridDim(_p.agents);
	if (testGridDim.x > 65535) {
		testGridDim.y = 1 + (testGridDim.x-1) / 65535;
		testGridDim.x = 1 + (testGridDim.x-1) / testGridDim.y;
	}

	// reset gradient kernel has total number of threads equal to the gradient values
	dim3 resetGradientBlockDim(512);
	dim3 resetGradientGridDim(1 + (_p.agents * _p.num_wgts - 1) / 512);
	if (resetGradientGridDim.x > 65535) {
		resetGradientGridDim.y = 1 + (resetGradientGridDim.x - 1) / 65535;
		resetGradientGridDim.x = 1 + (resetGradientGridDim.x - 1) / resetGradientGridDim.y;
	}
	
	// set up timing values
	CUDA_EVENT_PREPARE
	float timeReset = 0.0f;
	float timeLearn = 0.0f;
	float timeTest = 0.0f;
	float timeReduce = 0.0f;
	unsigned timerGPU;
	CREATE_TIMER(&timerGPU);
	START_TIMER(timerGPU);

	
	timing_feedback_header(_p.num_chunks);
	for (int i = 0; i < _p.num_chunks; i++) {
		timing_feedback_dot(i);
		
		unsigned isRestart = ((i > 0) && 0 == (i % _p.chunks_per_restart));
		if (isRestart) {
			// reset gradients (state will be randomized in the learning kernel)
			CUDA_EVENT_START;
			reset_gradient_kernel<<<resetGradientGridDim, resetGradientBlockDim>>>();
			CUDA_EVENT_STOP(timeReset);
			CUT_CHECK_ERROR("reset_gradient_kernel execution failed");
		}
		
		// do some learning
		CUDA_EVENT_START;
		learn_kernel<<<gridDim, blockDim>>>(_p.chunk_interval, isRestart);
		CUDA_EVENT_STOP(timeLearn);
		CUT_CHECK_ERROR("learn_kernel execution failed");
		
//		dump_agentsGPU("after learning session", agGPU);
		
		// run tests (pre-sharing)
		if (0 == ((i+1) % _p.chunks_per_test)) {
			CUDA_EVENT_START;
//			printf("test_kernel... (grid is (%d x %d) block is (%d x %d), index is %d\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y, ((i+1) / _p.chunks_per_test));
//			device_dumpf("test results, prior to test_kernel", d_results, _p.num_tests, _p.agents);
//			test_kernel<<<gridDim, blockDim>>>(d_results + ((i+1) / _p.chunks_per_test) * _p.agents);
//			printf("test_kernel2... (grid is (%d x %d) block is (%d x %d), index is %d\n", testGridDim.x, testGridDim.y, testBlockDim.x, testBlockDim.y, ((i+1) / _p.chunks_per_test));
			test_kernel2<<<testGridDim, testBlockDim>>>(d_results + ((i+1) / _p.chunks_per_test) * _p.agents);
			CUDA_EVENT_STOP(timeTest);
			CUT_CHECK_ERROR("test_kernel execution failed");
//			device_dumpf("test results, after test_kernel", d_results, _p.num_tests, _p.agents);
		}
		
//		dump_agentsGPU("after testing", agGPU);
		
		// sharing will go here
	}
	printf("\n");
	
	// reduce the result array on the device and copy back to host
	CUDA_EVENT_START;
	float *d_minval;
	unsigned *d_mincol;
//	device_dumpf("test results, prior to reduction", d_results, _p.num_tests, _p.agents);
	unsigned min_stride = row_argmin(d_results, _p.agents, _p.num_tests, &d_minval, &d_mincol);
//	printf("min_stride = %d\n", min_stride);
//	device_dumpf("d_minval", d_minval, _p.num_tests, 1 + (_p.agents - 1)/(2*BLOCK_SIZE));
//	device_dumpui("d_minval", d_mincol, _p.num_tests, 1 + (_p.agents - 1)/(2*BLOCK_SIZE));
	row_reduce(d_results, _p.agents, _p.num_tests);
	for (int i = 0; i < _p.num_tests; i++) {
		CUDA_SAFE_CALL(cudaMemcpy(rGPU->avg_fitness + i, d_results + i * _p.agents, sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(rGPU->best_fitness + i, d_minval + i * min_stride, sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(rGPU->best_agent + i, d_mincol + i * min_stride, sizeof(unsigned), cudaMemcpyDeviceToHost));
		rGPU->avg_fitness[i] /= _p.agents;
	}
	CUDA_EVENT_STOP(timeReduce);
	CUDA_EVENT_CLEANUP;
	STOP_TIMER(timerGPU, "total GPU time");
	PRINT_TIME(timeReset, "reset_gradient_kernel time");
	PRINT_TIME(timeLearn, "learn_kernel time");
	PRINT_TIME(timeTest, "test_kernel time");
	PRINT_TIME(timeReduce, "reduce_kernel time");

#ifdef DUMP_TERMINAL_AGENT_STATE
	dump_agents_GPU("--------------------------------------\n       Ending Agent States\n", agGPU);
#endif

	if (d_results) cudaFree(d_results);
}
