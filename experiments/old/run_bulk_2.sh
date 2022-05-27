#!/bin/bash

echo "running 1."
CUDA_VISIBLE_DEVICES=0 nohup python -um experiments.paper.train.train_er40 --save_loc data/ER40 --name run1 > ER40_run1.out &
BACK_PID_0a=$!

echo "running 2."
CUDA_VISIBLE_DEVICES=0 nohup python -um experiments.paper.train.train_er40 --save_loc data/ER40 --name run2 > ER40_run2.out &
BACK_PID_0b=$!

echo "running 3."
CUDA_VISIBLE_DEVICES=0 nohup python -um experiments.paper.train.train_er40 --save_loc data/ER40 --name run3 > ER40_run3.out &
BACK_PID_0c=$!

echo "running 4."
CUDA_VISIBLE_DEVICES=1 nohup python -um experiments.paper.train.train_ba40 --save_loc data/BA40 --name run1 > BA40_run1.out &
BACK_PID_1a=$!

echo "running 5."
CUDA_VISIBLE_DEVICES=1 nohup python -um experiments.paper.train.train_ba40 --save_loc data/BA40 --name run2 > BA40_run2.out &
BACK_PID_1b=$!

echo "running 6."
CUDA_VISIBLE_DEVICES=1 nohup python -um experiments.paper.train.train_ba40 --save_loc data/BA40 --name run3 > BA40_run3.out &
BACK_PID_1c=$!


wait $BACK_PID_0a
wait $BACK_PID_0b
wait $BACK_PID_0c

wait $BACK_PID_1a
wait $BACK_PID_1b
wait $BACK_PID_1c

echo "testing run_1's."

CUDA_VISIBLE_DEVICES=0 python -um experiments.paper.test.test --save_loc data/ER40 --name run1 --checkpoint_name solver_best_mean_ER500 -sd
BACK_PID_0=$!
CUDA_VISIBLE_DEVICES=1 python -um experiments.paper.test.test --save_loc data/BA40 --name run1 --checkpoint_name solver_best_mean_BA500 -sd
BACK_PID_1=$!

wait $BACK_PID_0
wait $BACK_PID_1

echo "testing run_2's."

CUDA_VISIBLE_DEVICES=0 python -um experiments.paper.test.test --save_loc data/ER40 --name run2 --checkpoint_name solver_best_mean_ER500 -sd
BACK_PID_0=$!
CUDA_VISIBLE_DEVICES=1 python -um experiments.paper.test.test --save_loc data/BA40 --name run2 --checkpoint_name solver_best_mean_BA500 -sd
BACK_PID_1=$!

wait $BACK_PID_0
wait $BACK_PID_1

echo "testing run_3's."

CUDA_VISIBLE_DEVICES=0 python -um experiments.paper.test.test --save_loc data/ER40 --name run3 --checkpoint_name solver_best_mean_ER500 -sd
BACK_PID_0=$!
CUDA_VISIBLE_DEVICES=1 python -um experiments.paper.test.test --save_loc data/BA40 --name run3 --checkpoint_name solver_best_mean_BA500 -sd
BACK_PID_1=$!

wait $BACK_PID_0
wait $BACK_PID_1

echo "done"