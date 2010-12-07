#!/bin/bash
# Basline testing script for resonance computer
#
#   ./res_fit_sharealways.sh <agent_group_size> <share_always_pct>
#
_time="--TIME_STEPS=1048576"
_groups="--AGENT_GROUP_SIZE=$1"
_restart="--RESTART_INTERVAL=64"
_test="--TEST_INTERVAL=256 --TEST_REPS=32 --TEST_MAX=250"
_share="--SHARE_BEST_PCT=0.50 --SHARE_ALWAYS_PCT=$2 --SHARE_FITNESS --COPY_ALPHA_MULTIPLIER=0.2"
_theta="--INIT_THETA_MIN=-0.10 --INIT_THETA_MAX=0.10"

_params="--ALPHA=0.20 --EPSILON=0.10 --GAMMA=0.95 --LAMBDA=0.2"

_location="./bin/linux/release"

_h="--HIDDEN_NODES=1"

_common="$_time $_groups $_restart $_test $_share $_theta"

$_location/mcar $_common $_params $_h --DUMP_BEST --DUMP_UPDATES --SEED=0
$_location/mcar $_common $_params $_h --DUMP_BEST --DUMP_UPDATES --SEED=1
$_location/mcar $_common $_params $_h --DUMP_BEST --DUMP_UPDATES --SEED=2
$_location/mcar $_common $_params $_h --DUMP_BEST --DUMP_UPDATES --SEED=3
$_location/mcar $_common $_params $_h --DUMP_BEST --DUMP_UPDATES --SEED=4
$_location/mcar $_common $_params $_h --DUMP_BEST --DUMP_UPDATES --SEED=5
$_location/mcar $_common $_params $_h --DUMP_BEST --DUMP_UPDATES --SEED=6
$_location/mcar $_common $_params $_h --DUMP_BEST --DUMP_UPDATES --SEED=7
