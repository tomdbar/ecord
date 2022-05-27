import argparse
from collections import defaultdict

import numpy as np

from experiments.configs import NetworkConfig, DQNTrainingConfig
from ecord.environments import NodeObservation, GlobalObservation
from ecord.graphs.generators import ErdosRenyiGraphGenerator, EdgeType
from ecord.models.rnn_decoder import RNNOutput, GRUDecoder
from ecord.models.ecodqn import GNN_ECODQN
from ecord.solvers.dqn import DQNSolver
from ecord.validate.testing import TestGraphsConfig, test_solver
from experiments.utils.system import save_to_disk

def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_loc",
        help="File to save experiment.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--num_steps",
        help="Number of steps per test.",
        type=int,
        default=100
    )
    parser.add_argument(
        "--num_tradj",
        help="Number of tradjectories per graph.",
        type=int,
        default=1
    )
    parser.add_argument(
        "--num_graphs",
        help="Number of graphs per test.",
        type=int,
        default=1
    )
    parser.add_argument(
        "--graph_size",
        help="Number of graphs per test.",
        type=int,
        default=None
    )
    parser.add_argument(
        "--ecodqn",
        help="Use ECO-DQN agent.",
        action='store_true',
        default=False
    )
    parser.add_argument(
        "--edge_density",
        help="edge density for ER graphs",
        type=float,
        default=0
    )
    args = parser.parse_args()
    return args


import torch
from torch_scatter import scatter_max
import time

def test_time(n):
    times = []
    for _ in range(10):
        x = torch.randn((n, 1)).float()
        b = torch.LongTensor([0] * len(x)).long()
        t = time.time()
        _ = scatter_max(x, b, dim=0)
        times.append(time.time() - t)

    print(f"Time : {np.mean(times) * 10 ** 3:.3f}ms")

    times = []
    for _ in range(10):
        x = torch.randn((n, 1)).float().cuda()
        b = torch.LongTensor([0] * len(x)).long().cuda()
        # print("x : shape, dtype, device", x.shape, x.dtype, x.device)
        # print("b : shape, dtype, device", b.shape, b.dtype, b.device)
        t = time.time()
        action_vals, actions = scatter_max(x, b, dim=0)
        times.append(time.time() - t)

    print(f"Time (cuda) : {np.mean(times) * 10 ** 3:.3f}ms")


if __name__ == "__main__":

    args = get_parser_args()

    if args.graph_size is not None:
        test_graph_sizes = [args.graph_size]
    else:
        test_graph_sizes = [10, 30, 100, 300, 1000, 3000, 10000]

    if not args.ecodqn:
        # ECORD config.

        network_config = NetworkConfig(
            rnn_model=GRUDecoder,

            # Features
            add_normalised_degree=False,
            add_degree_profile=False,
            add_weight_profile=False,
            norm_init_node_features=False,
            add_laplacian_PEs=False,

            node_features=[
                NodeObservation.STATE,
                NodeObservation.PEEK_NORM,
                NodeObservation.STEPS_STATIC_CLIP,
                # NodeObservation.ID,
                # NodeObservation.STATE_BEST,
            ],
            glob_features=[
                GlobalObservation.SCORE_FROM_BEST_NORM,
                # GlobalObservation.STEPS_SINCE_BEST_CLIP,
                # GlobalObservation.NUM_GREEDY_ACTIONS_CLIP,
                # GlobalObservation.STEPS_NORM,
            ],

            # GNN params.
            num_layers=4,
            dim_embedding=16,
            use_layer_norm=True,

            # GNN2RNN params
            dim_node_out=16,

            # Init decoding parameters
            use_network_initialisation=True,
            num_inital_probabilty_maps=2,

            # RNN params.
            num_rnn_layers=1,
            dim_features_embedding=16,
            add_glob_to_nodes=False,
            add_glob_to_obs=True,
            dim_internal_embedding=1024,
            learn_scale=False,
            learn_graph_embedding=False,
            project_rnn_to_gnn=True,
            output_type=RNNOutput.MLP_ACTION_STATE,
            output_activation=None,
        )

    else:
        # ECO-DQN config.

        network_config = NetworkConfig(
            gnn_model=GNN_ECODQN,

            # Features
            add_normalised_degree=False,
            add_degree_profile=False,
            add_weight_profile=False,
            norm_init_node_features=False,
            add_laplacian_PEs=False,

            node_features=[
                NodeObservation.STATE,
                NodeObservation.PEEK_NORM,
                NodeObservation.STEPS_STATIC_NORM,
                NodeObservation.STATE_BEST,
            ],
            glob_features=[
                GlobalObservation.SCORE_FROM_BEST_NORM,
                GlobalObservation.STEPS_SINCE_BEST_NORM,
                GlobalObservation.NUM_GREEDY_ACTIONS_CLIP,
                GlobalObservation.STEPS_NORM,
            ],

            # GNN params.
            num_layers=3,
            dim_embedding=64,
            add_glob_to_nodes=True,
            add_glob_to_obs=False,
        )

    def get_timing_stats(log, k):
        data = log[k]
        return np.array([data.mean(), data.std()]) * 10**3

    def create_sover():
        gnn = network_config.get_gnn()
        rnn = network_config.get_rnn()
        gnn2rnn = network_config.get_gnn2rnn()
        node_encoder = network_config.get_node_encoder()
        node_classifier = network_config.get_node_classifier()
        graph_batcher = network_config.get_graph_batcher()

        # Make config with defaults, we won't be training anyway...
        training_config = DQNTrainingConfig()

        solver = DQNSolver(
            gnn,
            rnn,
            gnn2rnn,
            node_encoder,
            node_classifier,

            node_features=network_config.node_features,
            global_features=network_config.glob_features,

            add_glob_to_nodes=network_config.add_glob_to_nodes,
            add_glob_to_obs=network_config.add_glob_to_obs,

            use_double_q_networks=False,
            use_ecodqn=args.ecodqn,

            device=None
        )
        solver.test()

        return solver

    res = defaultdict(list)

    def test_timings(
            solver,
            graphs,
            graph_batcher,
            num_steps,
            num_tradj,
            verbose=False,
    ):
        t = time.time()
        test_cfg = TestGraphsConfig(
            graphs=graphs,
            opt_scores=None,
            num_steps=num_steps,
            tradj_per_graph=num_tradj,
            use_network_initialisation=False,
            tau=0,
            fname=None,
            label=f'{n}')
        if verbose:
            print(f"...prep: {time.time() - t:.3f}s")

        t = time.time()
        log = test_solver(
            solver=solver,
            graph_batcher=graph_batcher,
            config=test_cfg,
            log_stats=False,
            verbose=False
        )
        t_test = time.time() - t
        return log, t_test


    num_par = args.num_graphs * args.num_tradj

    for n in test_graph_sizes:
    # for n in [10000]:

        print(f"\n\nGraph size = {n}\n\n")

        solver = create_sover()
        batcher = network_config.get_graph_batcher()
        gg = ErdosRenyiGraphGenerator([n,n], args.edge_density, EdgeType.SIGNED)
        graphs = [G for G in gg.generate(args.num_graphs)]

        max_tradj = None
        num_tradj = 275

        while max_tradj is None:
            # if num_tradj > 100:
            #     max_tradj = 100
            try:
                print(f"trying {num_tradj} tradj", end="...")
                test_timings(
                    solver,
                    graphs,
                    batcher,
                    1,
                    num_tradj,
                    verbose=False,
                )
                num_tradj += 1
                print("success.")
            except RuntimeError as e:
                print("OOM.")
                # max_tradj = -1
                print(f"OOM with batch size of {num_tradj}.")
                # del e
                raise e

        # Delete everything and wait for the GPU to clear.
        del solver, batcher, gg, graphs
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
        time.sleep(5)

        solver = create_sover()
        batcher = network_config.get_graph_batcher()
        gg = ErdosRenyiGraphGenerator([n, n], args.edge_density, EdgeType.SIGNED)
        graphs = [G for G in gg.generate(args.num_graphs)]

        log, t_test = test_timings(
            solver,
            graphs,
            batcher,
            args.num_steps,
            num_tradj,
            verbose = True,
        )

        step_timing = get_timing_stats(log, 't_step')
        inference_timing = get_timing_stats(log, 't_inf')
        calc_q_timing = get_timing_stats(log, 'calc_q_vals')
        action_select_timing = get_timing_stats(log, 'action_select')

        def get_log_line(label, timing, new_line = True):
            log_str = "\n            " if new_line else ""
            log_str += f"{label} --> {timing[0]:.3f}({timing[1]:.3f})ms"
            if num_par > 1:
                log_str += f", {label} per instance --> {timing[0]/num_par:.3f}({timing[1]/num_par:.3f})ms"
            return log_str

        log_str = f"size = {n}: "
        log_str += f"total solver time (exl. env set-up) : {log['tot_solver_time']:.3f}s"
        log_str += "\n            "
        log_str += f"set up --> {log['t_setup'].mean():.3f}s"
        if 't_init_emb' in log:
            log_str += f", gnn_embed --> {log['t_init_emb'].mean():.3f}s"
        log_str += f", rollout --> {log['t_rollout'].mean():.3f}s"
        log_str += get_log_line("inference+step", step_timing)
        log_str += get_log_line("inference only", inference_timing)
        log_str += get_log_line("q vals", calc_q_timing)
        log_str += get_log_line("action_select", action_select_timing)
        log_str += f"\n            total test time: {t_test:.3f}s"

        print(log_str)

        res['num_nodes'].append(n)
        res['num_tradj'].append(max_tradj)
        res['embedding_time'].append(log['t_init_emb'])
        res['step_time'].append(log['t_step'])
        res['inference_time'].append(log['t_inf'])
        res['calc_q_vals'].append(log['calc_q_vals'])
        res['action_select'].append(log['action_select'])

    if args.save_loc:
        save_to_disk(args.save_loc, res, compressed=True, verbose=True)