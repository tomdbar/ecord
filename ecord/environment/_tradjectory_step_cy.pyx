import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel cimport prange

@cython.boundscheck(False)  # turn off bounds-checking
@cython.wraparound(False)  # turn off negative index wrapping
@cython.cdivision(True)
cdef void _step(cnp.ndarray[cnp.int64_t, ndim=2] actions,
                cnp.ndarray[cnp.float32_t, ndim=2] scores,
                cnp.ndarray[cnp.float32_t, ndim=2] best_scores,
                cnp.ndarray[cnp.float32_t, ndim=2] best_states,

                # feature arrays
                cnp.ndarray[cnp.float32_t, ndim=3] node_features,
                cnp.ndarray[cnp.float32_t, ndim=3] global_features,

                # indices of feature arrays
                int node_state_dim,
                int node_peek_dim,
                int node_state_best_dim,
                int node_static_dim,
                int node_degree_dim,

                int glob_score_from_best_dim,
                int glob_steps_since_best_dim,
                int glob_num_greedy_actions_available_dim,

                # batch information
                cnp.ndarray[cnp.int32_t, ndim=2] edges_by_node,
                cnp.ndarray[cnp.float32_t, ndim=1] edge_attrs_by_node,
                cnp.ndarray[cnp.int32_t, ndim=1] edge_by_node_ptr,
                int num_nodes,
                int num_graphs,
                int num_tradj):
    '''
    actions : [num_graphs, num_traj]
    scores : [num_graphs, num_tradj]
    best_scores : [num_graphs, num_tradj]
    best_states : [num_nodes, num_tradj]

    node_features: [num_nodes, num_tradj, num_node_features]
    global_features: [num_graphs, num_tradj, num_glob_features]

    node_state_dim: int = 1
    node_peek_dim: int = 2
    node_static_dim: int
    node_state_best_dim: int
    node_degree_dim: int

    glob_score_from_best_dim: int
    glob_steps_since_best_dim: int
    glob_num_greedy_actions_available_dim: int

    edges_by_node : [2, num_nodes]
    edge_attrs_by_node : [num_nodes]
    edges_and_attrs_by_node_ptr : [num_nodes]
    degree : [num_nodes]
    num_nodes : int
    num_graphs : int
    num_tradj : int
    '''

    cdef Py_ssize_t i, j, k, k_max, act_idx
    cdef cnp.float32_t signed_weight
    cdef int new_best, delta_greedy
    cdef float new_peek

    with nogil:
        for i in prange(num_graphs):
            for j in range(num_tradj):
                '''
                For each action, update (in this order):
                    1. Score.
                    2. Best score of tradjectory.
                    3. Global score from best of tradjectory.
                    4. Global steps since best score of tradjectory.
                    5. Node peeks.
                    6. Global number of greed actions available.
                    7. Node states.
                    8. Node states at best.
                    9. Node steps static.
                '''
                act_idx = actions[i, j]

                # 1. Global score.
                scores[i, j] = scores[i, j] + node_features[act_idx, j, node_peek_dim]

                # 2. Global best score of tradjectory.
                new_best = 0
                if scores[i, j] > best_scores[i, j]:
                    best_scores[i, j] = scores[i, j]
                    new_best = 1

                # 3. Global score from best of tradjectory.
                if glob_score_from_best_dim >= 0:
                    global_features[i, j, glob_score_from_best_dim] = best_scores[i, j] - scores[i, j]

                # 4. Global steps since best score of tradjectory.
                if glob_steps_since_best_dim >= 0:
                    if new_best > 0:
                        global_features[i, j, glob_steps_since_best_dim] = 0
                    else:
                        global_features[i, j, glob_steps_since_best_dim] = global_features[
                                                                               i, j, glob_steps_since_best_dim] + 1

                # 5. Node peeks + 6. Global number of greed actions available.
                #    (i) Re-taking an action will reverse it's change.
                node_features[act_idx, j, node_peek_dim] = -1 * node_features[act_idx, j, node_peek_dim]
                if glob_num_greedy_actions_available_dim >= 0:
                    if node_features[act_idx, j, node_peek_dim] > 0:
                        global_features[i, j, glob_num_greedy_actions_available_dim] = \
                            global_features[i, j, glob_num_greedy_actions_available_dim] + 1
                    if node_features[act_idx, j, node_peek_dim] < 0:
                        global_features[i, j, glob_num_greedy_actions_available_dim] = \
                            global_features[i, j, glob_num_greedy_actions_available_dim] - 1

                #    (ii) Neighbouring nodes will gain/lose the connecting edge weight in cut-value.
                k, k_max = edge_by_node_ptr[act_idx], edge_by_node_ptr[act_idx + 1]
                while k < k_max:
                    signed_weight = (2 * node_features[edges_by_node[0, k], j, node_state_dim] - 1) * \
                                    (2 * node_features[edges_by_node[1, k], j, node_state_dim] - 1) * \
                                    edge_attrs_by_node[k]
                    new_peek = node_features[edges_by_node[1, k], j, node_peek_dim] - 2 * signed_weight

                    if glob_num_greedy_actions_available_dim >= 0:
                        delta_greedy = 0
                        if (new_peek > 0) and (node_features[edges_by_node[1, k], j, node_peek_dim] <= 0):
                            delta_greedy = 1
                        elif (new_peek <= 0) and (node_features[edges_by_node[1, k], j, node_peek_dim] > 0):
                            delta_greedy = -1
                        global_features[i, j, glob_num_greedy_actions_available_dim] = \
                            global_features[i, j, glob_num_greedy_actions_available_dim] + delta_greedy

                    node_features[edges_by_node[1, k], j, node_peek_dim] = new_peek

                    k = k + 1

                # 7. Node states.
                node_features[act_idx, j, node_state_dim] = (node_features[act_idx, j, node_state_dim] + 1) % 2

                # 8. Node states at best.
                if new_best > 0:
                    best_states[act_idx, j] = node_features[act_idx, j, node_state_dim]
                    if (node_state_best_dim > 0):
                        node_features[act_idx, j, node_state_best_dim] = best_states[act_idx, j]

                # 9. Node steps static.
                if node_static_dim >= 0:
                    node_features[act_idx, j, node_static_dim] = 0

def step_cy(actions, scores, best_scores, best_states,
            node_features, global_features,
            node_feature_idxs, glob_feature_idxs,
            edges_by_node, edge_attrs_by_node, edge_by_node_ptr,
            num_nodes, num_graphs, num_tradj):
    node_state_dim, node_peek_dim, node_state_best_dim, node_static_dim, node_degree_dim = node_feature_idxs
    glob_score_from_best_dim, glob_steps_since_best_dim, glob_num_greedy_actions_available_dim, glob_steps_dim, _, _ = glob_feature_idxs

    if node_static_dim >= 0:
        node_features[..., node_static_dim] += 1

    if glob_steps_dim >= 0:
        global_features[..., glob_steps_dim] += 1

    return _step(actions, scores, best_scores, best_states,
                 node_features, global_features,
                 node_state_dim, node_peek_dim, node_state_best_dim, node_static_dim, node_degree_dim,
                 glob_score_from_best_dim, glob_steps_since_best_dim, glob_num_greedy_actions_available_dim,
                 edges_by_node, edge_attrs_by_node, edge_by_node_ptr,
                 num_nodes, num_graphs, num_tradj)