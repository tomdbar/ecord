import os
import pickle
from dataclasses import dataclass, field, fields
from typing import List, Optional, Callable

import torch
import torch.nn as nn

from ecord.environment import Tradjectories, NodeObservation, GlobalObservation
from ecord.environment.data import GraphBatcher
from ecord.graphs.generators import _GraphGeneratorBase, ErdosRenyiGraphGenerator, EdgeType
from ecord.models.ecodqn import GNN_ECODQN
from ecord.models.gnn_embedder import GNNEmbedder
from ecord.models.gnn_to_rnn import GNN2RNN
from ecord.models.node_classifier import NodeClassifier
from ecord.models.rnn_decoder import _RNNDecoderBase, RNNOutput, GRUDecoder


@dataclass
class NetworkConfig:
    # Models
    gnn_model: nn.Module = GNNEmbedder
    rnn_model: _RNNDecoderBase = GRUDecoder

    # Features
    add_laplacian_PEs: bool = False
    add_normalised_degree: bool = False
    add_degree_profile: bool = False
    add_weight_profile: bool = False
    norm_init_node_features: bool = False

    node_features: List[NodeObservation] = field(default_factory=lambda: [
        NodeObservation.STATE,
        NodeObservation.PEEK_NORM,
        #         NodeObservation.STEPS_STATIC_NORM,
        # NodeObservation.LAPLACIAN_PE # <-- doesn't (currently) play nicely with ragged batches.
    ])
    glob_features: List[GlobalObservation] = field(default_factory=lambda: [
        GlobalObservation.SCORE_FROM_BEST_NORM,
        #         GlobalObservation.STEPS_SINCE_BEST_NORM, # <-- seems to cause instability...
        GlobalObservation.NUM_GREEDY_ACTIONS_NORM,
        GlobalObservation.STEPS_NORM
    ])

    # GNN params
    dim_embedding: int = 16
    num_layers: int = 10
    use_layer_norm: bool = True
    return_dummy_graph_embeddings: Optional[bool] = False

    # GNN2NN params
    dim_node_out: int = 64

    # Init decoding parameters
    use_network_initialisation: bool = False
    num_inital_probabilty_maps: int = 1
    decode_with_init_state: bool = False

    # Node encoder (embeddings+features --> embeddings) params
    dim_features_embedding: int = None
    add_glob_to_nodes: bool = True
    add_glob_to_obs: bool = True

    # RNN params.
    num_rnn_layers: int = 1
    learn_scale: bool = False
    output_type: RNNOutput = RNNOutput.DOT
    output_activation: Callable[[torch.tensor], torch.tensor] = None
    dim_internal_embedding: Optional[int] = None
    learn_graph_embedding: Optional[bool] = False
    project_rnn_to_gnn: Optional[bool] = False
    return_dummy_hidden_state: Optional[bool] = False
    # add_node_features_to_last_action_embedding: Optional[bool] = False

    def __post_init__(self):
        if self.gnn_model is GNNEmbedder:
            self.use_ecodqn = False
        elif self.gnn_model is GNN_ECODQN:
            self.use_ecodqn = True
        else:
            raise Exception(f"Unsupported GNN configured: {self.gnn_model}.")

    def get_gnn_args(self):
        if not self.use_ecodqn:
            dim_in_nodes = 0
            if self.add_degree_profile:
                dim_in_nodes += 5
            if self.add_weight_profile:
                dim_in_nodes += 5
            if self.add_normalised_degree:
                dim_in_nodes += 1
            if self.add_laplacian_PEs:
                dim_in_nodes += 3

            gnn_args = {
                    'dim_in': max(1, dim_in_nodes),
                    'use_layer_norm': True,
                    'dim_embedding': max(self.dim_embedding, dim_in_nodes),
                    'num_layers': 4,
                    'return_dummy_graph_embeddings': self.return_dummy_graph_embeddings,
                    'device': None,
                    'out_device': None,
                }
        else:
            gnn_args = {
                'dim_in': self.get_dim_node_features(),
                'dim_embedding': self.dim_embedding,
                'num_layers': 3,
                'device': None,
                'out_device': None,
            }

        return gnn_args

    def get_gnn(self):
        gnn = self.gnn_model(
            **self.get_gnn_args()
        )
        return gnn

    def get_gnn2rnn(self):
        gnn2rnn = GNN2RNN(
            dim_node_in = self.get_gnn_args()['dim_embedding'],
            dim_node_out = self.dim_node_out,
            device=None,
            out_device=None,
        )
        return gnn2rnn

    def get_dim_node_features(self):
        if self.dim_features_embedding is not None and self.dim_features_embedding > 0:
            return self.dim_features_embedding
        else:
            dim = len(self.node_features)
            if self.add_glob_to_nodes:
                dim += len(self.glob_features)
            return dim

    def get_rnn_args(self):
        dim_node_feat = self.get_dim_node_features()
        if NodeObservation.LAPLACIAN_PE in self.node_features:
            dim_node_feat += (Tradjectories._lap_PEs_k - 1)

        dim_internal_embedding = self.dim_internal_embedding
        if dim_internal_embedding is None:
            dim_internal_embedding = self.dim_node_out

        dim_msg_in = self.dim_node_out + dim_node_feat
        if self.add_glob_to_obs:
            dim_msg_in += len(self.glob_features)

        rnn_args = {
            'dim_node_in': self.dim_node_out,
            'dim_msg_in': dim_msg_in,
            'dim_node_embedding': self.dim_node_out + dim_node_feat,
            # 'dim_node_embedding': self.get_dim_gnn_and_obs_embedding(),
            'dim_internal_embedding': dim_internal_embedding,
            'learn_graph_embedding': self.learn_graph_embedding,
            'project_rnn_to_gnn': self.project_rnn_to_gnn,
            'output_type': self.output_type,
            'output_activation': self.output_activation,
            'learn_scale': self.learn_scale,
            'num_layers': self.num_rnn_layers,
            'return_dummy_hidden_state': self.return_dummy_hidden_state,
        }
        return rnn_args

    # def get_dim_gnn_and_obs_embedding(self):
    #     if self.dim_features_embedding is not None and self.dim_features_embedding > 0:
    #         return self.dim_features_embedding
    #     else:
    #         return len(self.node_features) + self.dim_node_out

    def get_node_encoder(self):
        if self.dim_features_embedding is not None and self.dim_features_embedding > 0:
            dim = len(self.node_features)
            if NodeObservation.LAPLACIAN_PE in self.node_features:
                dim += (Tradjectories._lap_PEs_k - 1)

            # dim = len(self.node_features) + self.dim_node_out
            if self.add_glob_to_nodes:
                dim += len(self.glob_features)

            node_encoder = nn.Sequential(
                nn.Linear(dim, self.dim_features_embedding),
                # nn.LayerNorm(self.dim_features_embedding),
                # nn.LeakyReLU()
            )

        else:
            node_encoder = None
        return node_encoder

    def get_node_classifier(self):
        if self.use_network_initialisation:
            # node_classifier = nn.Sequential(
            #     nn.Linear(self.dim_features_embedding, self.num_inital_probabilty_maps),
            #     # nn.Sigmoid(),
            # )
            node_classifier = NodeClassifier(
                self.get_gnn_args()['dim_embedding'],
                num_probabilty_maps=self.num_inital_probabilty_maps,
            )
        else:
            node_classifier = None
        return node_classifier

    def get_rnn(self):
        return self.rnn_model(**self.get_rnn_args())

    def get_graph_batcher(self):
        return GraphBatcher(
            add_normalised_degree=self.add_normalised_degree,
            add_degree_profile=self.add_degree_profile,
            add_weight_profile=self.add_weight_profile,
            norm_features=self.norm_init_node_features,
            add_laplacian_PEs = self.add_laplacian_PEs,
        )

    def save_to_file(self, fname='network', quiet=False):
        if os.path.splitext(fname)[-1] != ".config":
            fname += ".config"
        if not quiet:
            print(f"Saving NetworkConfig to {fname}", end="...")
        with open(fname, 'wb') as file:
            pickle.dump(self, file)
        if not quiet:
            print("done.")

    def __repr__(self):
        str = f"---{self.__class__.__name__}---\n"
        for field in fields(self):
            attr_str = f"{getattr(self, field.name)}"
            if len(attr_str) > 78:
                attr_str = attr_str[:75] + "..."
            str += f"\n\t{field.name} = {attr_str}"
        return str + "\n"


@dataclass
class DQNTrainingConfig:
    # Episode
    graph_generator: _GraphGeneratorBase = ErdosRenyiGraphGenerator([40, 40], 0.15, EdgeType.SIGNED)
    tradj_per_graph: int = 1
    num_steps_per_rollout: int = -2
    num_parallel_graphs: int = 8
    allow_reversible_actions: bool = True
    intermediate_reward: float = 0
    revisit_reward: float = 0

    # Training
    num_steps: int = 1000
    lr: float = 5e-4
    gamma: float = 0.9

    update_target_frequency: int = 1
    update_target_polyak: int = 0.01
    use_double_q_networks: bool = False
    use_gnn_embeddings_from_buffer: bool = False
    default_actor_idx: int = None
    tau: float = 0
    alpha: float = 0
    munch_lower_log_clip: float = -1

    init_bce_weight: float = 0
    detach_gnn_from_rnn: bool = False
    prob_network_initialisation: float = 0

    batch_size: int = 32
    num_sub_batches: int = 2
    env_steps_per_update: int = 1

    initial_epsilon: float = 1.
    final_epsilon: float = 0.05
    final_epsilon_step: int = 1000

    # Replay buffer.
    buffer_capacity: int = 5000
    buffer_add_probability: float = 1
    k_BPTT: int = 5
    n_step: int = 1
    min_buffer_length: int = 500
    crop_tradjectories_at_final_reward: bool = False

    # Testing
    test_freq: int = 5

    # System
    save_loc: str = './'
    checkpoint_freq: int = None
    save_final: bool = True

    def __repr__(self):
        str = f"---{self.__class__.__name__}---\n"
        for field in fields(self):
            attr_str = f"{getattr(self, field.name)}"
            if len(attr_str) > 78:
                attr_str = attr_str[:75] + "..."
            str += f"\n\t{field.name} = {attr_str}"
        return str + "\n"