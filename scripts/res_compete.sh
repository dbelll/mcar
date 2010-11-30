#!/bin/bash
# Testing script for resonance computer
_trials="--TRIALS=1"
_time="--TIME_STEPS=65536"
_groups="--AGENT_GROUP_SIZE=32"
_interval="--SHARING_INTERVAL=1024"
_restart="--RESTART_INTERVAL=1024"

_run="--RUN_ON_CPU=0 --RUN_ON_GPU=1"
_location="./bin/linux/release"

_test="--TEST_INTERVAL=1024 --TEST_REPS=128 --TEST_MAX=500"
_theta="--INIT_THETA_MIN=-0.10 --INIT_THETA_MAX=0.10"

_params="--ALPHA=0.20 --EPSILON=0.10 --GAMMA=0.95 --LAMBDA=0.2"
_h="--HIDDEN_NODES=1"

_common="$_trials $_time $_groups $_interval $_restart $_run $_test $_theta"

$_location/mcar $_common $_params $_h --DUMP_BEST


