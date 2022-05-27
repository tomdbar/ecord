import argparse
import os
import pickle
from datetime import datetime
import time

from ecord.solvers import DQNSolver
from ecord.validate.testing import TestGraphsConfig, test_solver
from experiments.configs import DQNTrainingConfig
from experiments.utils.system import mk_dir, export_summary, save_to_disk


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_loc",
        help="Folder of saved experiment.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--name",
        help="Experiment name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--checkpoint_name",
        help="Folder of saved experiment.",
        type=str,
        default='checkpoints/final_solver',
    )
    parser.add_argument(
        "--id",
        help="String to identify this experiment.",
        type=str,
        default='',
    )
    parser.add_argument(
        "--graph_loc",
        help="Location of GSet graph",
        type=str,
        default='graphs/ecodqn/validation/ER_40spin_p15_100graphs.pkl',
    )
    parser.add_argument(
        "--num_steps",
        help="Steps per tradjectory",
        type=int,
        default=-2,
    )
    parser.add_argument(
        "--max_time",
        help="Maximum solver time (default is no limit)",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--num_tradj",
        help="Number of tradjectories per graph",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--tau",
        help="Temperature of policy",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--num_load",
        help="Number of graphs to load.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--graph_batch_size",
        help="Number of graphs per batch.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--tradj_batch_size",
        help="Number of trajectories per batch.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--pre_solve",
        help="Whether to pre solve with greedy.",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--post_solve",
        help="Whether to post solve with greedy from the best state.",
        action='store_true',
        default=False,
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
    parser.add_argument(
        "--ecodqn",
        help="Use ECO-DQN agent.",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--log_stats", '-l',
        help="Whether to log detailed statistics (this is slower).",
        action='store_true',
        default=False,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_parser_args()
    now = datetime.now()
    save_loc = args.save_loc
    if args.save_loc is None:
        save_loc = f"data/{now.strftime('%y_%m_%d')}"
    exp_name = args.name
    if exp_name is None:
        exp_name = now.strftime("%H_%M_%S")
    exp_dir = os.path.join(save_loc, exp_name)
    info_dir = os.path.join(exp_dir, "info")
    test_dir = os.path.join(exp_dir, f"test/{args.checkpoint_name.replace('/', '_')}")

    if not os.path.isdir(exp_dir):
        raise Exception(f"Target directory ({exp_dir}) doesn't exist.")

    network_config_fname = os.path.join(info_dir, "network.config")

    print(f"Loading network config from {network_config_fname}", end="...")
    with open(network_config_fname, 'rb') as network_config_file:
        network_config = pickle.load(network_config_file)
    print("done.\n")

    gnn = network_config.get_gnn()
    rnn = network_config.get_rnn()
    gnn2rnn = network_config.get_gnn2rnn()
    node_encoder = network_config.get_node_encoder()
    node_classifier = network_config.get_node_classifier()
    graph_batcher = network_config.get_graph_batcher()

    # Make config with defaults, we won't be training anyway...
    training_config = DQNTrainingConfig()

    checkpoint_name = args.checkpoint_name
    if os.path.splitext(checkpoint_name) != ".pth":
        checkpoint_name += ".pth"

    solver_checkpoint_fname = os.path.join(exp_dir, checkpoint_name)

    solver_args = {
        'gnn':gnn,
        'rnn':rnn,
        'gnn2rnn':gnn2rnn,
        'node_encoder':node_encoder,
        'node_classifier':node_classifier,

        'use_ecodqn': args.ecodqn,

        'add_glob_to_nodes': network_config.add_glob_to_nodes,
        'add_glob_to_obs': network_config.add_glob_to_obs,

        'node_features': network_config.node_features,
        'global_features': network_config.glob_features,

        'allow_reversible_actions' : training_config.allow_reversible_actions,
        'default_actor_idx' : None,
        'device' : None
    }

    try:
        # Try double Q-network
        solver = DQNSolver(
            use_double_q_networks=True,
            **solver_args,
        )
        solver.load(solver_checkpoint_fname, quiet=False)
    except:
        # Try single Q-network
        solver = DQNSolver(
            use_double_q_networks=False,
            **solver_args,
        )
        solver.load(solver_checkpoint_fname, quiet=False)

    solver.test()

    graph_dir, graph_name = os.path.split(args.graph_loc)
    if graph_dir=='':
        graph_dir = '_gset'

    test_config = TestGraphsConfig.from_file(
        label=graph_name,
        graph_loc=os.path.join(graph_dir, graph_name),
        num_steps=args.num_steps,
        tradj_per_graph=args.num_tradj,
        tau=args.tau,
        max_time=args.max_time,
        pre_solve_with_greedy=args.pre_solve,
        post_solve_with_greedy_from_best=args.post_solve,
        use_network_initialisation=False,
        num_load=args.num_load,
        graphs_bsz=args.graph_batch_size,
        tradj_bsz=args.tradj_batch_size,
    )

    if args.dump_logs or args.save_summary:
        mk_dir(test_dir, quiet=True)

    summary_str = ""
    res_str = "\n--- Results summary ---\n"

    _summary_str = test_config.get_summary()
    summary_str += f"{_summary_str}\n"
    print(_summary_str)

    t = time.time()
    log = test_solver(
        solver=solver,
        graph_batcher=graph_batcher,
        config=test_config,
        log_stats=args.log_stats,
        verbose=True
    )
    t_test = time.time() - t

    lab = test_config.get_label()
    if args.dump_logs:
        t = time.time()
        fname = os.path.join(test_dir, lab)
        if args.id != '':
            fname += f"_{args.id}"
        save_to_disk(fname, log, compressed=True, verbose=True)
        t_dump = time.time() - t

    scores_max, scores_mean = log['apx_max'], log['apx_mean']
    if scores_max is None:
        scores_max, scores_mean = log['scores_max'], log['scores_mean']
    t_step = log['t_step'].mean() * 10 ** 3

    tmp_res_str = f"\n{lab}:"
    tmp_res_str += f"\n\tBest (mean) score - raw : {scores_max.mean():.3f} ({scores_mean.mean():.3f}) - {log['scores_max'].mean().item()}."
    tmp_res_str += f"\n\tAve. time to opt: {log['t_opt'].mean():.3f}s out of {log['t_tot'].mean():.3f}s."
    tmp_res_str += f"\n\tAve. batched step time: {t_step:.3f}ms."
    if args.dump_logs:
        tmp_res_str += f"\n\tTotal testing/saving time: {t_test:.1f}/{t_dump:.1f}s.\n"

    print(tmp_res_str)

    res_str += tmp_res_str

    if args.save_summary:
        summary_fname = os.path.join(test_dir, f"summary")
        if args.id != '':
            summary_fname += f"_{args.id}"
        summary_fname += ".txt"
        print(f"Saving summary to {summary_fname}", end="...")
        export_summary(
            summary_fname,
            summary_str + res_str
        )
        print("done.")

    print(res_str)