/*
 *  main.h
 *  mcar
 *
 *  Created by Dwight Bell on 11/20/10.
 *  Copyright 2010 dbelll. All rights reserved.
 *
 */


//#define VERBOSE

//#define DUMP_INITIAL_AGENTS
#define DUMP_FINAL_AGENTS

//#define DEBUG_CALC_Q
//#define DEBUG_THETA_UPDATE
//#define DEBUG_CPU
//#define DUMP_STATES
//#define DUMP_AGENT_UPDATES

//#define TRACE_TEST

#ifdef DEBUG_CPU
#define DUAL_PREFIX __host__
#else
#define DUAL_PREFIX __host__ __device__
#endif
