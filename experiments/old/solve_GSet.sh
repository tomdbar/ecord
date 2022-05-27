#!/bin/bash

while getopts d:s:n:c:g:x:t:l: flag
do
    case "${flag}" in
        d) device=${OPTARG};;
    esac
done
echo "device : $device";

#for idx in 1 2 3 4 5 43 44 45 46 47 22 23 24 25 26 55 60 70
for idx in 1 2 3 4 5 43 44 45 46 47 22 23 24 25 26

do
  echo "running G$idx"
#  CUDA_VISIBLE_DEVICES=$device python -um experiments.paper.test.test_large --save_loc data/ablations/ER500_DQN --name run1 --checkpoint_name solver_best_mean_ER10000 --num_steps -100000 --num_tradj 20 --max_time 180 --tau 0 --graph_loc G$idx -sd
  CUDA_VISIBLE_DEVICES=$device python -um experiments.paper.test.test_large --save_loc data/ablations/ER500_noGNN --name run1 --checkpoint_name solver_best_mean_ER10000 --num_steps -100000 --num_tradj 20 --max_time 180 --tau 0.0 --graph_loc G$idx -sd
  CUDA_VISIBLE_DEVICES=$device python -um experiments.paper.test.test_large --save_loc data/ablations/ER500_noRNN --name run1 --checkpoint_name solver_best_mean_ER10000 --num_steps -100000 --num_tradj 20 --max_time 180 --tau 0.0005 --graph_loc G$idx -sd
done

#CUDA_VISIBLE_DEVICES=$device python -um experiments.paper.test.test_large --save_loc data/ablations/ER500_noGNN --name run1 --checkpoint_name solver_best_mean_ER10000 --num_steps -100000 --num_tradj 20 --max_time 180 --tau 0.00035 --graph_loc G55 -sd
#CUDA_VISIBLE_DEVICES=$device python -um experiments.paper.test.test_large --save_loc data/ablations/ER500_noGNN --name run1 --checkpoint_name solver_best_mean_ER10000 --num_steps -100000 --num_tradj 20 --max_time 180 --tau 0.0002 --graph_loc G60 -sd
#CUDA_VISIBLE_DEVICES=$device python -um experiments.paper.test.test_large --save_loc data/ablations/ER500_noGNN --name run1 --checkpoint_name solver_best_mean_ER10000 --num_steps -100000 --num_tradj 20 --max_time 180 --tau 0.0001 --graph_loc G70 -sd