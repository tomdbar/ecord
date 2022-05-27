import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import networkx as nx
import torch

from ecord.environment.data import GraphBatcher
from ecord.graphs.utils import load_graph_from_file
from ecord.solvers import DQNSolver


@dataclass()
class TestGraphsConfig:
    graphs: List[nx.Graph]
    opt_scores: List[int]
    num_steps: int
    tradj_per_graph: int
    act_greedy: bool = False
    use_network_initialisation: bool = False
    tau: float = None
    max_time: float = None, # In seconds
    pre_solve_with_greedy: bool = False
    post_solve_with_greedy_from_best: bool = False
    graphs_bsz: int = None
    tradj_bsz: int = None
    fname: str = None
    label: Optional[str] = None

    def __post_init__(self):
        self.num_graphs = len(self.graphs)

        if self.graphs_bsz is None or self.graphs_bsz >= self.num_graphs:
            self.__num_graph_batches = 1
            self.graphs_bsz = self.num_graphs
        else:
            self.__num_graph_batches = math.ceil(self.num_graphs / self.graphs_bsz)

        if self.tradj_bsz is None or self.tradj_bsz >= self.tradj_per_graph:
            self.__num_tradj_batches = 1
            self.tradj_bsz = self.tradj_per_graph
        else:
            self.__num_tradj_batches = math.ceil(self.tradj_per_graph / self.tradj_bsz)

    @staticmethod
    def from_file(
            graph_loc,
            num_load=None,
            idx_load=None,  # or (i,j) to load graphs[i:j] or [i,j,k,...] to load graphs[i], graphs[j] etc.
            num_steps=-1,
            tradj_per_graph=1,
            act_greedy=False,
            use_network_initialisation=False,
            tau=None,
            max_time=None,
            post_solve_with_greedy_from_best=False,
            pre_solve_with_greedy=False,
            graphs_bsz=None,
            tradj_bsz=None,
            label=None,
    ):
        graphs, opt_scores = load_graph_from_file(graph_loc, quiet=True)
        if num_load is not None and idx_load is not None:
            print("Warning: only one of num_load or idx_load should be specified..." +
                  "defaulting to using num_load.")
        elif num_load is not None:
            graphs = graphs[:num_load]
            if opt_scores:
                opt_scores = opt_scores[:num_load]
        elif isinstance(idx_load, List) and len(idx_load == 2):
            graphs, opt_scores = graphs[idx_load[0]:idx_load[1]]
            if opt_scores:
                opt_scores = opt_scores[idx_load[0]:idx_load[1]]
        else:
            graphs, opt_scores = [graphs[i] for i in idx_load]
            if opt_scores:
                opt_scores = [opt_scores[i] for i in idx_load]
        if opt_scores:
            opt_scores = torch.tensor(opt_scores)

        return TestGraphsConfig(
            graphs=graphs,
            opt_scores=opt_scores,
            num_steps=num_steps,
            tradj_per_graph=tradj_per_graph,
            act_greedy=act_greedy,
            use_network_initialisation=use_network_initialisation,
            tau=tau,
            max_time=max_time,
            pre_solve_with_greedy=pre_solve_with_greedy,
            post_solve_with_greedy_from_best=post_solve_with_greedy_from_best,
            graphs_bsz=graphs_bsz,
            tradj_bsz=tradj_bsz,
            fname=graph_loc,
            label=label,
        )

    def get_num_batches(self):
        return self.__num_graph_batches, self.__num_tradj_batches

    def get_summary(self):
        summary_str = f"Testing {self.num_graphs} graphs "
        if self.fname is not None:
            summary_str += f"from {self.fname}, "
        summary_str += f"with {self.graphs_bsz} graphs and {self.tradj_bsz} tradjectories per rollout."
        return summary_str

    def get_label(self):
        if self.label is not None:
            return self.label
        elif self.fname is not None:
            return os.path.split(self.fname)[-1]
        else:
            return "test_graphs"


def merge_log(log_src, log_trg, dim=-1):
    for k, v in log_src.items():
        v_trg = log_trg.get(k, None)
        if isinstance(v[0], torch.Tensor):
            # scores/actions/peeks: [graphs, tradj, steps]
            if not isinstance(v, torch.Tensor):
                v = torch.stack(v, dim=-1)
            if v_trg is None:
                log_trg[k] = v
            else:
                if v_trg.dim() > dim and  v.dim() > dim:
                    log_trg[k] = torch.cat([v_trg, v], dim=dim)
                else:
                    # If in doubt, just cat along leading axis.
                    log_trg[k] = torch.cat([v_trg, v], dim=0)
        else:
            # t_step: [steps]
            v = torch.tensor(v)
            if v_trg is None:
                log_trg[k] = v
            else:
                log_trg[k] += v

    return log_trg


@torch.no_grad()
def test_solver(
        solver: DQNSolver,
        graph_batcher: GraphBatcher,
        config: TestGraphsConfig,
        log_stats: bool = False,
        verbose: bool = False,
):
    num_g_batches, num_t_batches = config.get_num_batches()
    log = defaultdict()

    was_training = solver.is_training
    solver.test()

    for i in range(num_g_batches):
        batch = graph_batcher(
            config.graphs[i * config.graphs_bsz: min((i + 1) * config.graphs_bsz, config.num_graphs)]
        )

        log_batch = defaultdict()
        tot_tradj = 0
        tot_solver_time = 0
        for j in range(num_t_batches):
            num_tradj = min(config.tradj_bsz, config.tradj_per_graph - tot_tradj)

            if i > 0 or j > 0:
                # Tidy memory fragmentation if doing multiple loops.
                torch.cuda.empty_cache()

            if verbose:
                print(f"\t...rollout {i * num_t_batches + j + 1}/{num_g_batches * num_t_batches}", end="...")
            t = time.time()
            _log_rollout = solver.rollout(
                batch,
                num_tradj=config.tradj_bsz,
                num_steps=config.num_steps,
                log_stats=log_stats,
                use_epsilon=False,
                use_network_initialisation=config.use_network_initialisation,
                tau=config.tau,
                max_time=config.max_time,
                pre_solve_with_greedy=config.pre_solve_with_greedy,
                post_solve_with_greedy_from_best=config.post_solve_with_greedy_from_best,
            )
            tot_solver_time += (time.time() - t) - torch.cat(_log_rollout['t_setup']).mean()
            if verbose:
                print("done", end="\r")

            log_batch = merge_log(_log_rollout, log_batch, dim=1)
            tot_tradj += num_tradj

            solver.actor.reset_state()

        log = merge_log(log_batch, log, dim=0)

    scores_by_tradj, score_idx_by_tradj = log['scores'].max(-1)
    scores_max, scores_mean, scores_idx_mean = (scores_by_tradj.max(-1).values,
                                                scores_by_tradj.mean(-1),
                                                score_idx_by_tradj.float().mean(-1))
    init_score_mean = log['init_score'].mean(-1)

    log['scores_max'] = scores_max
    log['scores_mean'] = scores_mean
    log['scores_idx_mean'] = scores_idx_mean
    log['init_score_mean'] = init_score_mean
    log['tot_solver_time'] = tot_solver_time

    log['num_graphs'] = config.num_graphs
    log['num_tradj'] = config.tradj_per_graph
    if config.opt_scores is not None:
        opt_scores = config.opt_scores
        log['opt'] = opt_scores
        log['apx_max'] = scores_max / opt_scores
        log['apx_mean'] = scores_mean / opt_scores
        log['init_apx_score_mean'] = init_score_mean / opt_scores

    else:
        log['apx_max'] = None
        log['apx_mean'] = None
        log['init_apx_score_mean'] = None

    if was_training:
        solver.train()

    return log