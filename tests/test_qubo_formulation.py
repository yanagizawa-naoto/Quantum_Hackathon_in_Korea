"""
Unit tests to verify the QUBO formulation in optimization_service_small_world.

The QUBO uses a unified reward convention: -1 * indicator(path exists) is added
to Q for every reachable directed pair (d1 and d2). Because the sampler *minimises*
Q, lower values correspond to more reachable pairs, which is the desired behaviour.

Variable encoding for canonical edge (u, v) with u < v:
    x_{u}_{v} = 0  →  edge directed u -> v  (forward)
    x_{u}_{v} = 1  →  edge directed v -> u  (backward)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import defaultdict
from service.optimization_service_small_world import (
    _add_d1_path_reward,
    _add_d2_path_reward,
    _build_qubo_for_small_world,
    _prepare_graph_data,
)


def _eval_qubo(Q, assignment):
    """Evaluate the QUBO objective x^T Q x for a given variable assignment."""
    total = 0.0
    for (vi, vj), coeff in Q.items():
        total += coeff * assignment.get(vi, 0) * assignment.get(vj, 0)
    return total


class TestD1PathReward:
    """Tests for _add_d1_path_reward sign convention."""

    def test_reward_forward_direction(self):
        """For i < j, x=0 (i->j) should score lower than x=1 (j->i)."""
        Q = defaultdict(float)
        edge_to_var = {(0, 1): "x_0_1"}
        _add_d1_path_reward(Q, 0, 1, edge_to_var)  # path 0->1, i < j
        # x=0 (0->1 path exists): contribution = Q[(x_0_1,x_0_1)] * 0 = 0
        # x=1 (0->1 path absent): contribution = Q[(x_0_1,x_0_1)] * 1 = +1
        assert Q[("x_0_1", "x_0_1")] == 1.0
        assert _eval_qubo(Q, {"x_0_1": 0}) < _eval_qubo(Q, {"x_0_1": 1})

    def test_reward_backward_direction(self):
        """For i > j, x=1 (i->j) should score lower than x=0 (j->i)."""
        Q = defaultdict(float)
        edge_to_var = {(0, 1): "x_0_1"}
        _add_d1_path_reward(Q, 1, 0, edge_to_var)  # path 1->0, i > j
        # x=1 (1->0 path exists): contribution = Q[(x_0_1,x_0_1)] * 1 = -1
        # x=0 (1->0 path absent): contribution = 0
        assert Q[("x_0_1", "x_0_1")] == -1.0
        assert _eval_qubo(Q, {"x_0_1": 1}) < _eval_qubo(Q, {"x_0_1": 0})

    def test_both_directions_cancel(self):
        """
        Calling for both (i,j) and (j,i) should yield net zero QUBO contribution.
        This is correct: each edge always provides exactly 1 d1-reachable pair,
        regardless of orientation, so d1 alone cannot distinguish orientations.
        """
        Q = defaultdict(float)
        edge_to_var = {(0, 1): "x_0_1"}
        _add_d1_path_reward(Q, 0, 1, edge_to_var)
        _add_d1_path_reward(Q, 1, 0, edge_to_var)
        assert Q[("x_0_1", "x_0_1")] == 0.0

    def test_missing_edge_is_skipped(self):
        """No QUBO entry should be added when the edge does not exist."""
        Q = defaultdict(float)
        edge_to_var = {(0, 1): "x_0_1"}
        _add_d1_path_reward(Q, 0, 2, edge_to_var)  # edge (0,2) absent
        assert len(Q) == 0


class TestD2PathReward:
    """Tests for _add_d2_path_reward sign convention."""

    def test_reward_two_hop_path(self):
        """
        For the path 0->1->2 on a 3-node chain, the QUBO should score
        lower when both edges are forward (x=0) than when one is reversed.
        """
        Q = defaultdict(float)
        edge_to_var = {(0, 1): "x_0_1", (1, 2): "x_1_2"}
        _add_d2_path_reward(Q, 0, 2, 1, edge_to_var)  # path 0->1->2

        # Both edges forward: path 0->1->2 exists
        qubo_path_exists = _eval_qubo(Q, {"x_0_1": 0, "x_1_2": 0})
        # First edge reversed: path absent
        qubo_no_path = _eval_qubo(Q, {"x_0_1": 1, "x_1_2": 0})
        assert qubo_path_exists < qubo_no_path

    def test_consistent_negative_reward_sign(self):
        """
        The quadratic coefficient (quad_coeff) should be negative when both
        edge indicators share the same sign, rewarding co-direction of the path.
        For 0->1->2 (i < k < j): coeff_ik = -1, coeff_kj = -1, quad = -((-1)(-1)) = -1.
        """
        Q = defaultdict(float)
        edge_to_var = {(0, 1): "x_0_1", (1, 2): "x_1_2"}
        _add_d2_path_reward(Q, 0, 2, 1, edge_to_var)
        assert Q[("x_0_1", "x_1_2")] == -1.0

    def test_reward_reverse_two_hop_path(self):
        """
        For the path 2->1->0 (reverse direction), the QUBO should score
        lower when both edges are backward (x=1) than when one is forward.
        """
        Q = defaultdict(float)
        edge_to_var = {(0, 1): "x_0_1", (1, 2): "x_1_2"}
        _add_d2_path_reward(Q, 2, 0, 1, edge_to_var)  # path 2->1->0

        qubo_path_exists = _eval_qubo(Q, {"x_0_1": 1, "x_1_2": 1})
        qubo_no_path = _eval_qubo(Q, {"x_0_1": 0, "x_1_2": 1})
        assert qubo_path_exists < qubo_no_path


class TestBuildQuboSmallWorld:
    """Integration tests for _build_qubo_for_small_world."""

    def _qubo_for_graph(self, vertices, edges):
        canonical_edges, vertex_set, adj = _prepare_graph_data(edges, vertices)
        return _build_qubo_for_small_world(vertex_set, adj, canonical_edges), canonical_edges

    def test_three_node_chain_prefers_aligned_orientations(self):
        """
        For a 3-node chain 0-1-2, orientations that create a 2-hop path
        (0->1->2 or 2->1->0) should have strictly lower QUBO than orientations
        that block all 2-hop paths.
        """
        Q, _ = self._qubo_for_graph([0, 1, 2], [[0, 1], [1, 2]])

        # x_0_1=0, x_1_2=0 → 0->1->2: 2-hop path exists
        aligned_fwd = _eval_qubo(Q, {"x_0_1": 0, "x_1_2": 0})
        # x_0_1=1, x_1_2=1 → 2->1->0: 2-hop path exists
        aligned_rev = _eval_qubo(Q, {"x_0_1": 1, "x_1_2": 1})
        # x_0_1=0, x_1_2=1 → 0->1, 2->1: no 2-hop path
        diverging = _eval_qubo(Q, {"x_0_1": 0, "x_1_2": 1})
        # x_0_1=1, x_1_2=0 → 1->0, 1->2: no 2-hop path
        converging = _eval_qubo(Q, {"x_0_1": 1, "x_1_2": 0})

        assert aligned_fwd < diverging
        assert aligned_fwd < converging
        assert aligned_rev < diverging
        assert aligned_rev < converging

    def test_no_edges_returns_empty_qubo(self):
        """A graph with no edges should produce an empty QUBO."""
        canonical_edges, vertex_set, adj = _prepare_graph_data([], [0, 1, 2])
        Q = _build_qubo_for_small_world(vertex_set, adj, canonical_edges)
        assert len(Q) == 0

    def test_single_edge_qubo_is_zero(self):
        """
        A single edge has two orientations, each providing the same number of
        reachable pairs (1 d1 pair). No orientation is preferred, so the QUBO
        should evaluate to the same value for both.
        """
        Q, _ = self._qubo_for_graph([0, 1], [[0, 1]])
        assert _eval_qubo(Q, {"x_0_1": 0}) == _eval_qubo(Q, {"x_0_1": 1})
