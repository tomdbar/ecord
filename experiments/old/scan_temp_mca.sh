#!/bin/bash

while getopts t:d:n: flag
do
    case "${flag}" in
        t) max_time=${OPTARG};;
        d) device=${OPTARG};;
        n) num_steps=${OPTARG};;
    esac
done
echo "max_time : $max_time";
echo "device : $device";
echo "device : $num_steps";

#for graph_loc in G1 G43 G22 G55 G60 G70
for graph_loc in  G1 G2 G3 G4 G5 G43 G44 G45 G46 G47 G22 G23 G24 G25 G26 G55 G60 G70
#for graph_loc in  G1 G2 G3 G4 G5

do
#  for tau in 0.000001 0.000003 0.00001 0.00003 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3 1 3
  for tau in 0 0.0001 0.001 0.01 0.1 1

  do
    echo "running $graph_loc, tau=$tau"
#    CUDA_VISIBLE_DEVICES=$device python -um experiments.paper.test.test_gset_mca --num_steps -1000000 --num_tradj 20 --tau $tau --max_time $max_time --graph_loc $graph_loc
#    CUDA_VISIBLE_DEVICES=$device python -um experiments.paper.test.test_gset_mca --num_steps $num_steps --num_tradj 20 --tau $tau --max_time $max_time --save_loc data/mca_soft/${graph_loc}_tau${tau} --graph_loc $graph_loc
    CUDA_VISIBLE_DEVICES=$device python -um experiments.paper.test.test_gset_mca --num_tradj 20 --tau $tau --max_time $max_time --save_loc data/mca_soft/${graph_loc}_tau${tau} --graph_loc $graph_loc
#    CUDA_VISIBLE_DEVICES=$device python -um experiments.paper.test.test_gset_mca --num_tradj 20 --tau 0 --save_loc data/mca_soft/${graph_loc}_tau0 --graph_loc $graph_loc

  done

done