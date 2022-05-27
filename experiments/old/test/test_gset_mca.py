import argparse
import os

from experiments.utils.system import save_to_disk
from ecord.environments.data import GraphBatcher
from ecord.environment.environment import Environment
from ecord.solvers.solver import greedy_solve
from ecord.validate.testing import TestGraphsConfig
import torch


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph_loc",
        help="Location of GSet graph",
        type=str,
        default='_gset/G1',
    )
    parser.add_argument(
        "--num_steps",
        help="Steps per tradjectory",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--num_tradj",
        help="Number of tradjectories per graph",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_time",
        help="Maximum solver time",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--tau",
        help="MCA temp",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--save_loc",
        help="Folder of saved experiment.",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_parser_args()

    graph_dir, graph_name = os.path.split(args.graph_loc)
    if graph_dir == '':
        graph_dir = '_gset'

    batcher = GraphBatcher()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Running on", device)

    test_config = TestGraphsConfig.from_file(
        label=graph_name,
        graph_loc=os.path.join(graph_dir, graph_name),
        max_time=args.max_time,
        num_steps=args.num_steps,
        tradj_per_graph=args.num_tradj,
        tau=args.tau,
        use_network_initialisation=False,
        num_load=1,
    )

    batch = batcher(test_config.graphs)
    env = Environment(
        batch.nx,
        test_config.tradj_per_graph,
        batch.tg,
        device=device
    )
    out = greedy_solve(
        env,
        tau=test_config.tau,
        max_steps=args.num_steps,
        max_time=args.max_time,
        ret_log=args.save_loc is not None,
        verbose=True,
        device=device
    )
    if args.save_loc:
        env, log = out
        save_to_disk(args.save_loc, log, compressed=True, verbose=True)
    else:
        env = out

    scores = env.tradj.get_best_scores()
    if test_config.opt_scores is not None:
        scores = env.tradj.get_best_scores() / test_config.opt_scores.unsqueeze(1)

    scores_max, scores_mean = scores.max(-1).values,  scores.mean(-1)

    print(f"Best (mean) score : {scores_max.mean():.3f} ({scores_mean.mean():.3f})")