#!/bin/bash

while getopts d:s:n:c:g:x:t:l: flag
do
    case "${flag}" in
        d) device=${OPTARG};;
        s) save_loc=${OPTARG};;
        n) name=${OPTARG};;
        c) checkpoint_name=${OPTARG};;
        g) graph_loc=${OPTARG};;
        x) num_steps=${OPTARG};;
        t) num_tradj=${OPTARG};;
        l) num_load=${OPTARG};;
    esac
done
echo "device : $device";
echo "save_loc: $save_loc";
echo "name: $name";
echo "checkpoint_name : $checkpoint_name";
echo "graph_loc : $graph_loc";
echo "num_steps : $num_steps";
echo "num_tradj : $num_tradj";
echo "num_load : $num_load";

#for tau in 0 0.000001 0.000003 0.000006 0.00001 0.00003 0.00006 0.0001 0.0003 0.0006 0.001
#for tau in 0 0.00000001 0.0000001 0.000001 0.00001 0.0001 0.001 0.01 0.1 1

#for tau in 0.00002 0.00003 0.00004 0.00005 0.00006 0.00007 0.00008 0.00009 # ER10000/ER7000/ER5000
#for tau in 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.0009 # ER10000/ER7000/ER5000
#for tau in 0.00011 0.00012 0.00013 0.00014 0.00015 0.00016 0.00017 0.00018 00019 # ER10000
#for tau in 0.000125 0.00015 0.000175 0.000225 0.00025 0.000275 0.000325 0.00035 0.000375 0.000425 0.00045 0.000475 # ER5000/ER7000

for tau in 0.00002 0.00003 0.00004 0.00005

do
  echo "running tau=$tau"
  CUDA_VISIBLE_DEVICES=$device python -um experiments.paper.test.test_large --save_loc $save_loc --name $name --checkpoint_name $checkpoint_name --num_steps $num_steps --num_tradj $num_tradj --num_load $num_load --tau $tau --graph_loc $graph_loc --id "tau_$tau" -sd
done