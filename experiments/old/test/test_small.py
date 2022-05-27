import argparse
import os
import pickle
from datetime import datetime
import time

from ecord.solvers.dqn import DQNSolver
from ecord.validate.testing import TestGraphsConfig, test_solver
from experiments.configs import DQNTrainingConfig
from experiments.utils.system import mk_dir, export_summary, save_to_disk


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        help="Experiment name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_loc",
        help="Folder of saved experiment.",
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
        "--log_stats", '-l',
        help="Whether to log detailed statistics (this is slower).",
        action='store_true',
        default=False,
    )
    parser.add_argument(
        "--ecodqn",
        help="Use ECO-DQN agent.",
        action='store_true',
        default=False
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
    test_dir = os.path.join(exp_dir, f"test/{args.checkpoint_name.replace('/','_')}")

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
        # Try doulbe Q-network
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

    def get_graph_loc(num_nodes, graph_type='ER'):
        graph_type = graph_type.upper()
        if graph_type == 'ER':
            return f'_graphs_eco_dqn/validation/ER_{num_nodes}spin_p15_100graphs'
        elif graph_type == 'BA':
            return f'_graphs_eco_dqn/validation/BA_{num_nodes}spin_m4_100graphs'
        elif graph_type == 'GSET':
            if num_nodes not in [800, 2000]:
                raise ValueError(f"num_nodes = {num_nodes} is not valid for Gset graphs.")
            return f'_graphs_eco_dqn/benchmarks/gset_{num_nodes}spin_graphs'
        else:
            raise ValueError

    test_config_args = {
        'num_steps': -2,
        'tradj_per_graph': 50,
        'num_load': 100,
        'use_network_initialisation': False,
        'tau': 0,
    }

    test_graph_configs = []

    test_graph_configs += [
        TestGraphsConfig.from_file(
            label=f'ER40',
            graph_loc=get_graph_loc(40,'ER'),
            **test_config_args,
        ),
        TestGraphsConfig.from_file(
            label=f'ER60',
            graph_loc=get_graph_loc(60, 'ER'),
            **test_config_args,
        ),
        TestGraphsConfig.from_file(
            label=f'ER100',
            graph_loc=get_graph_loc(100, 'ER'),
            graphs_bsz = None if not args.ecodqn else 10,
            **test_config_args,
        ),
        TestGraphsConfig.from_file(
            label=f'ER200',
            graph_loc=get_graph_loc(200, 'ER'),
            graphs_bsz=None if not args.ecodqn else 5,
            **test_config_args,
        ),
        TestGraphsConfig.from_file(
            label=f'ER500',
            graph_loc=get_graph_loc(500, 'ER'),
            graphs_bsz=None if not args.ecodqn else 1,
            tradj_bsz=None if not args.ecodqn else 10,
            **test_config_args,
        )
    ]

    test_graph_configs += [
        TestGraphsConfig.from_file(
            label=f'BA40',
            graph_loc=get_graph_loc(40,'BA'),
            **test_config_args,
        ),
        TestGraphsConfig.from_file(
            label=f'BA60',
            graph_loc=get_graph_loc(60, 'BA'),
            **test_config_args,
        ),
        TestGraphsConfig.from_file(
            label=f'BA100',
            graph_loc=get_graph_loc(100, 'BA'),
            graphs_bsz=None if not args.ecodqn else 10,
            **test_config_args,
        ),
        TestGraphsConfig.from_file(
            label=f'BA200',
            graph_loc=get_graph_loc(200, 'BA'),
            graphs_bsz=None if not args.ecodqn else 5,
            **test_config_args,
        ),
        TestGraphsConfig.from_file(
            label=f'BA500',
            graph_loc=get_graph_loc(500, 'BA'),
            graphs_bsz=None if not args.ecodqn else 1,
            tradj_bsz=None if not args.ecodqn else 10,
            **test_config_args,
        )
    ]

    test_graph_configs += [
        # TestGraphsConfig.from_file(
        #     label=f'G1-10 50step1',
        #     graph_loc=get_graph_loc(800, 'GSet'),
        #     num_steps=-1,
        #     tradj_per_graph=50,
        #     # tradj_bsz=50,
        #     num_load=10,
        #     use_network_initialisation=False,
        #     tau=0),
        # TestGraphsConfig.from_file(
        #     label=f'G1-10 50step2',
        #     graph_loc=get_graph_loc(800, 'GSet'),
        #     num_steps=-2,
        #     tradj_per_graph=50,
        #     # tradj_bsz=50,
        #     num_load=10,
        #     use_network_initialisation=False,
        #     tau=0),
        # TestGraphsConfig.from_file(
        #     label=f'G1-10 50step4',
        #     graph_loc=get_graph_loc(800, 'GSet'),
        #     num_steps=-4,
        #     tradj_per_graph=50,
        #     # tradj_bsz=50,
        #     num_load=10,
        #     use_network_initialisation=False,
        #     tau=0),

        # TestGraphsConfig.from_file(
        #     label=f'G1-10 50step8',
        #     graph_loc=get_graph_loc(800, 'GSet'),
        #     num_steps=-8,
        #     tradj_per_graph=50,
        #     # tradj_bsz=50,
        #     num_load=10,
        #     use_network_initialisation=False,
        #     tau=0),
        # TestGraphsConfig.from_file(
        #     label=f'G1-10 50step16',
        #     graph_loc=get_graph_loc(800, 'GSet'),
        #     num_steps=-16,
        #     tradj_per_graph=50,
        #     # tradj_bsz=50,
        #     num_load=10,
        #     use_network_initialisation=False,
        #     tau=0),
        # TestGraphsConfig.from_file(
        #     label=f'G1-10 50step32',
        #     graph_loc=get_graph_loc(800, 'GSet'),
        #     num_steps=-32,
        #     tradj_per_graph=50,
        #     # tradj_bsz=50,
        #     num_load=10,
        #     use_network_initialisation=False,
        #     tau=0),
        #
        # TestGraphsConfig.from_file(
        #     label=f'G22-32',
        #     graph_loc=get_graph_loc(2000, 'GSet'),
        #     num_steps=-2,
        #     tradj_per_graph=50,
        #     num_load=10,
        #     use_network_initialisation=False,
        #     tau=0),
    ]

    if args.dump_logs or args.save_summary:
        mk_dir(test_dir, quiet=True)

    summary_str = ""
    res_str = "\n--- Results summary ---\n"
    for config in test_graph_configs:
        _summary_str = config.get_summary()
        summary_str += f"{_summary_str}\n"
        print(_summary_str)

        t = time.time()
        log = test_solver(
            solver=solver,
            graph_batcher=graph_batcher,
            config=config,
            log_stats=args.log_stats,
            verbose=True
        )
        t_test = time.time() - t

        lab = config.get_label()
        if args.dump_logs:
            t = time.time()
            fname = os.path.join(test_dir, lab)
            save_to_disk(fname, log, compressed=True, verbose=True)
            t_dump = time.time() - t

        scores_max, scores_mean = log['apx_max'], log['apx_mean']
        if scores_max is None:
            scores_max, scores_mean = log['scores_max'], log['scores_mean']
        t_step = log['t_step'].mean() * 10 ** 3

        tmp_res_str = f"\n{lab}:"
        tmp_res_str += f"\n\tBest (mean) score : {scores_max.mean():.3f} ({scores_mean.mean():.3f})."
        tmp_res_str += f"\n\tAve. batched step time: {t_step:.3f}ms."
        tmp_res_str += f"\n\tTotal testing/saving time: {t_test:.1f}/{t_dump:.1f}s.\n"

        print(tmp_res_str)

        res_str += tmp_res_str

    if args.save_summary:
        summary_fname = os.path.join(test_dir, f"summary.txt")
        print(f"Saving summary to {summary_fname}", end="...")
        export_summary(
            summary_fname,
            summary_str + res_str
        )
        print("done.")

    print(res_str)