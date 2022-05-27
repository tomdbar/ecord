#!/bin/bash

while getopts d:s:n:c:g:x:t:l: flag
do
    case "${flag}" in
        d) device=${OPTARG};;
    esac
done
echo "device : $device";

for idx in 1 2 3 4 5 43 44 45 46 47 22 23 24 25 26

do
  echo "running G$idx"
  python -um experiments.paper.test.test_gset_mca --num_steps -1000000 --num_tradj 20 --tau 1 --max_time 180 --graph_loc G$idx
done

#CUDA_VISIBLE_DEVICES=$device python -um experiments.paper.test.test_large --save_loc data/ER500_bin --name run3 --checkpoint_name solver_best_mean_ER10000_p0005_bin --num_steps -100000 --num_tradj 20 --max_time 300 --tau 0.00035 --graph_loc G55 -sd
#CUDA_VISIBLE_DEVICES=$device python -um experiments.paper.test.test_large --save_loc data/ER500_bin --name run3 --checkpoint_name solver_best_mean_ER10000_p0005_bin --num_steps -100000 --num_tradj 20 --max_time 300 --tau 0.0002 --graph_loc G60 -sd
#CUDA_VISIBLE_DEVICES=$device python -um experiments.paper.test.test_large --save_loc data/ER500_bin --name run3 --checkpoint_name solver_best_mean_ER10000_p0005_bin --num_steps -100000 --num_tradj 20 --max_time 300 --tau 0.0001 --graph_loc G70 -sd