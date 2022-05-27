#!/bin/bash

echo "running 1."
CUDA_VISIBLE_DEVICES=1 nohup python -um experiments.paper.train.ecodqn.train_er200_p12_binary --save_loc data/ER40_bin_ecodqn --name run1 > ER40_bin_ecodqn_1.out &
BACK_PID=$!
wait $BACK_PID

echo "running 2."
CUDA_VISIBLE_DEVICES=1 nohup python -um experiments.paper.train.ecodqn.train_er200_p12_binary --save_loc data/ER40_bin_ecodqn --name run2 > ER40_bin_ecodqn_2.out &
BACK_PID=$!
wait $BACK_PID

echo "running 3."
CUDA_VISIBLE_DEVICES=1 nohup python -um experiments.paper.train.ecodqn.train_er200_p12_binary --save_loc data/ER40_bin_ecodqn --name run2 > ER40_bin_ecodqn_3.out &
BACK_PID=$!
wait $BACK_PID

echo "done"