import argparse
import os
from datetime import datetime

import scipy as sp
import scipy.stats

from experiments.utils.system import mk_dir, export_summary, save_to_disk
from ecord.environments.data import GraphBatcher
from ecord.environment.environment import Environment
from ecord.solvers.solver import greedy_solve
from ecord.validate.testing import TestGraphsConfig


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_loc",
        help="Folder to save results.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_summary", '-s',
        help="Whether to save a summary.",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--dump_logs", '-d',
        help="Whether to save the logs.",
        action='store_true',
        default=False,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_parser_args()
    now = datetime.now()
    save_loc = args.save_loc
    if args.save_loc is None and (args.save_summary or args.dump_logs):
        save_loc = f"data/{now.strftime('%y_%m_%d')}/mca"

    if (args.save_summary or args.dump_logs):
        mk_dir(save_loc, quiet=False)

    batcher = GraphBatcher()

    test_config_args = {
        'num_steps': -2,
        'tradj_per_graph': 50,
        'num_load': 100,
    }

    def get_graph_loc(graph_type, n):
        if graph_type=='ER':
            loc = f'_graphs_eco_dqn/validation/ER_{n}spin_p15_100graphs'
        elif graph_type=='BA':
            loc = f'_graphs_eco_dqn/validation/BA_{n}spin_m4_100graphs'
        else:
            raise Exception("Unrecognised graph type.")
        return loc

    def get_confidence_bounds(scores):
        ae, loce, scalee = sp.stats.skewnorm.fit(scores)
        lb, ub = sp.stats.skewnorm.interval(0.68, ae, loce, scalee)
        return lb, ub

    sum_str = "\n--- Results summary ---\n"

    for graph_type in ['ER','BA']:
        for n in [40,60,100,200,500]:

            lab = f"\n{graph_type}{n}"
            res_str = lab

            res = {}
            for tau in [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]:
                cfg = TestGraphsConfig.from_file(
                    label=lab,
                    graph_loc=get_graph_loc(graph_type, n),
                    tau=tau,
                    use_network_initialisation=False,
                    **test_config_args,
                )
                batch = batcher(cfg.graphs)
                env = Environment(
                    batch.nx,
                    cfg.tradj_per_graph,
                    batch.tg,
                    device="cpu"
                )
                env = greedy_solve(env, max_steps=cfg.num_steps, tau=cfg.tau, ret_log=False)
                scores = env.tradj.get_best_scores()
                if cfg.opt_scores is not None:
                    scores = env.tradj.get_best_scores() / cfg.opt_scores.unsqueeze(1)
                res[tau] = scores

                # Printing results.
                scores_max = scores.max(1).values
                lb_max, up_max = get_confidence_bounds(scores_max)

                scores_mean = scores.mean(1)
                lb_mean, up_mean = get_confidence_bounds(scores_mean)

                def fmt_errs(s, ub, lb):
                    return f"{s:.3f}(+{ub-s:.3f},-{s-lb:.3f})"

                res_str += f"\n\ttau={tau} : {fmt_errs(scores_max.mean(), up_max, lb_max)}, {fmt_errs(scores_mean.mean(), up_mean, lb_mean)}"
                print(res_str)

            # Saving results.
            if args.dump_logs:
                fname = os.path.join(save_loc, lab)
                save_to_disk(fname, res, compressed=True, verbose=True)

            print(res_str)
            sum_str += res_str

    if args.save_summary:
        summary_fname = os.path.join(save_loc, f"summary.txt")
        print(f"Saving summary to {summary_fname}", end="...")
        export_summary(
            summary_fname,
            sum_str
        )
        print("done.")