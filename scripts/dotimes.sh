#!/bin/bash

# repeat a script with different seeds
# 1st argument is the script file (without the 'scripts/' prefix)
# 2nd argument is the number of times to repeat (should be <= 16)

for(( seed=0; seed<$2; seed++))
do

./scripts/$1 $seed

done

