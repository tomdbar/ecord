import argparse
import os
from datetime import datetime

from experiments.configs import NetworkConfig, DQNTrainingConfig
from experiments.utils.plotting import plot_training
from experiments.utils.system import export_script, mk_dir
from ecord.environments import NodeObservation, GlobalObservation
from ecord.graphs.generators import ErdosRenyiGraphGenerator, EdgeType
from ecord.models.rnn_decoder import RNNOutput, GRUDecoder
from ecord.solvers.dqn import DQNSolver, Trainer
from ecord.validate.testing import TestGraphsConfig


def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        help = "Experiment name",
        type=str,
        default=None
    )
    parser.add_argument(
        "--graph_type",
        help="Graph type",
        type=str,
        default='ER'
    )
    parser.add_argument(
        "--graph_size",
        help="Graph size",
        type=int,
        default=40
    )
    parser.add_argument(
        "--save_loc",
        help="Folder to save experiment.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--continue_exp", "-c",
        help="Whether to continue existing experiment if it exists.",
        action="store_true",
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

    mk_dir(exp_dir, quiet=True)
    mk_dir(info_dir, quiet=True)

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
            # GlobalObservation.NUM_GREEDY_ACTIONS_NORM,
            # GlobalObservation.STEPS_NORM,
            GlobalObservation.PEEK_MAX_NORM,
            # GlobalObservation.PEEK_MEAN_NORM,
        ],

        # GNN params.
        num_layers=4,
        dim_embedding=16,
        use_layer_norm=True,

        # GNN2RNN params
        dim_node_out=16,

        # Init decoding parameters
        use_network_initialisation=False,
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

    training_config = DQNTrainingConfig(
        # Episode
        graph_generator = ErdosRenyiGraphGenerator([500, 500], (0.05,0.0), EdgeType.CONSTANT),
        tradj_per_graph = 1,
        num_steps_per_rollout = -2,
        num_parallel_graphs = 8,
        intermediate_reward = 0,
        revisit_reward = 0,
        prob_network_initialisation=0,

        # Training
        num_steps = 40000,
        lr = 1e-3,
        gamma = 0.7,
        init_bce_weight=0e-4, # Only matters if network_config.use_network_initialisation is True.
        detach_gnn_from_rnn=False,

        update_target_frequency=1,
        update_target_polyak=0.01,
        use_double_q_networks=False,
        default_actor_idx=0,

        # Soft-DQN/M-DQN
        tau=0.0001,

        # M-DQN
        alpha=0.9,
        munch_lower_log_clip=-1,

        # Batch params
        batch_size = 64,
        num_sub_batches = 1,
        env_steps_per_update = 8,
        # batch_size=32,
        # num_sub_batches=1,
        # env_steps_per_update=1,

        initial_epsilon = 1,
        final_epsilon = 0.05,
        final_epsilon_step = 5000,

        # Replay buffer.
        buffer_capacity = 40000,
        buffer_add_probability = 1,
        k_BPTT = 5,
        n_step = 1,
        min_buffer_length = 20000,
        crop_tradjectories_at_final_reward = False,
        use_gnn_embeddings_from_buffer = False,

        # Testing
        test_freq = 500,

        # System
        save_loc = exp_dir,
        checkpoint_freq = None,
        save_final = True,
    )


    gnn = network_config.get_gnn()
    rnn = network_config.get_rnn()
    gnn2rnn = network_config.get_gnn2rnn()
    node_encoder = network_config.get_node_encoder()
    node_classifier = network_config.get_node_classifier()

    solver = DQNSolver(
        gnn,
        rnn,
        gnn2rnn,
        node_encoder,
        node_classifier,

        detach_gnn_from_rnn=training_config.detach_gnn_from_rnn,
        add_glob_to_nodes=network_config.add_glob_to_nodes,
        add_glob_to_obs=network_config.add_glob_to_obs,

        node_features=network_config.node_features,
        global_features=network_config.glob_features,

        allow_reversible_actions=training_config.allow_reversible_actions,
        intermediate_reward=training_config.intermediate_reward,
        revisit_reward=training_config.revisit_reward,

        initial_epsilon=training_config.initial_epsilon,
        final_epsilon=training_config.final_epsilon,
        final_epsilon_step=training_config.final_epsilon_step,

        use_double_q_networks=training_config.use_double_q_networks,
        use_gnn_embeddings_from_buffer=training_config.use_gnn_embeddings_from_buffer,
        default_actor_idx=training_config.default_actor_idx,
        tau=training_config.tau,
        alpha=training_config.alpha,
        munch_lower_log_clip=training_config.munch_lower_log_clip,

        device=None
    )

    test_config_args = {
        'num_steps': -2,
        'tradj_per_graph': 10,
        'use_network_initialisation': False,
        'tau': 0.0001,
        'graphs_bsz': None,
        'tradj_bsz': None,
    }

    test_graph_configs = [
        TestGraphsConfig.from_file(
            label='ER500',
            graph_loc='_graphs/binary/ER500',
            num_load=20,
            **test_config_args,
        ),
        TestGraphsConfig.from_file(
            # label='ER10000_p0005_bin',
            # graph_loc='_graphs/binary/ER_10000spin_p0005_5graphs',
            label='ER10000',
            graph_loc='_graphs/binary/ER10000',
            num_load=5,
            **test_config_args,
        ),
        ]

    graph_batcher = network_config.get_graph_batcher()

    trainer = Trainer(
        solver,

        graph_batcher=graph_batcher,

        training_gg=training_config.graph_generator,
        test_graph_configs = test_graph_configs,

        batch_size=training_config.batch_size,
        num_sub_batches=training_config.num_sub_batches,
        lr=training_config.lr,

        buffer_capacity=training_config.buffer_capacity,
        buffer_add_probability=training_config.buffer_add_probability,
        k_BPTT=training_config.k_BPTT,
        n_step=training_config.n_step,
        min_buffer_length=training_config.min_buffer_length,

        crop_tradjectories_at_final_reward = training_config.crop_tradjectories_at_final_reward,
        add_final_scores_to_transitions = network_config.use_network_initialisation,

        # Training
        gamma=training_config.gamma,
        init_bce_weight=training_config.init_bce_weight,
        prob_network_initialisation=training_config.prob_network_initialisation,

        num_parallel_graphs=training_config.num_parallel_graphs,
        tradj_per_graph=training_config.tradj_per_graph,
        num_steps_per_rollout=training_config.num_steps_per_rollout,
        update_target_frequency=training_config.update_target_frequency,
        update_target_polyak=training_config.update_target_polyak,

        save_loc=training_config.save_loc,
        checkpoint_freq=training_config.checkpoint_freq
    )

    if args.continue_exp:
        trainer.load()
    else:
        network_config.save_to_file(os.path.join(info_dir, "network.config"), quiet=False)

    export_script(__file__, info_dir)

    trainer.train(
        num_steps=training_config.num_steps,
        test_freq=training_config.test_freq,
        verbose=True,
        save_final=training_config.save_final,
    )

    fig = plot_training(
        trainer,
        test_set_labels=[config.get_label() for config in test_graph_configs],
        window=1,
        log_scale=False,
        plot_means=True,
        plot_solve_steps=True,
    )
    fig.savefig(
        os.path.join(trainer.save_loc, "scores.png"),
        bbox_inches='tight'
    )