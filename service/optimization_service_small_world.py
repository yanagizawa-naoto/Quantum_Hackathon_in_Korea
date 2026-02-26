from collections import defaultdict
import itertools
from typing import Dict, Set, Tuple, List
from dwave.samplers import SimulatedAnnealingSampler
from dto.ResponseDto import EdgeDto

# --- Private Helper Functions: Level 2 (Lowest Level Logic) ---

def _add_d1_path_reward(
    Q: Dict[Tuple[str, str], float], i: int, j: int, edge_to_var: Dict[Tuple[int, int], str]
):
    """
    Adds a reward term to the QUBO for a direct path (length 1) from i to j.

    Uses the unified reward convention: -1 * indicator(i->j exists) is added to Q so
    that minimising Q maximises the number of reachable ordered pairs.

    For canonical edge (u, v) with u < v, x=0 means u->v and x=1 means v->u.
    The indicator that i->j exists is (1 - x) when i < j, or x when i > j.
    Adding -1 * indicator gives:
        i < j:  -(1 - x) = x - 1  →  +1.0 on the diagonal
        i > j:  -x               →  -1.0 on the diagonal
    Both cases apply the same reward convention; the sign differs only because the
    indicator formula flips depending on whether i->j is the forward or reverse
    direction of the canonical edge variable.
    """
    edge_ij = tuple(sorted((i, j)))
    if edge_ij not in edge_to_var:
        return
    var_ij = edge_to_var[edge_ij]
    Q[(var_ij, var_ij)] += 1.0 if i < j else -1.0


def _add_d2_path_reward(
    Q: Dict[Tuple[str, str], float], i: int, j: int, k: int, edge_to_var: Dict[Tuple[int, int], str]
):
    """
    Adds a reward term to the QUBO for a 2-hop path (i -> k -> j).

    Uses the unified reward convention: -1 * indicator(i->k->j exists) is added to Q.
    The path indicator is indicator_ik * indicator_kj, where each edge indicator is a
    linear function of its variable: indicator = const + coeff * x.

    Expanding -indicator_ik * indicator_kj yields one quadratic term and two linear
    (diagonal) correction terms, all with consistent negative-reward signs.
    """
    edge_ik = tuple(sorted((i, k)))
    var_ik = edge_to_var[edge_ik]
    coeff_ik = 1.0 if i > k else -1.0
    const_ik = 0.0 if i > k else 1.0

    edge_kj = tuple(sorted((k, j)))
    var_kj = edge_to_var[edge_kj]
    coeff_kj = 1.0 if k > j else -1.0
    const_kj = 0.0 if k > j else 1.0

    # Quadratic term: -coeff_ik * coeff_kj * x_ik * x_kj
    quad_coeff = -(coeff_ik * coeff_kj)
    if var_ik < var_kj:
        Q[(var_ik, var_kj)] += quad_coeff
    else:
        Q[(var_kj, var_ik)] += quad_coeff

    # Linear correction terms from expanding the product
    Q[(var_ik, var_ik)] -= coeff_ik * const_kj
    Q[(var_kj, var_kj)] -= const_ik * coeff_kj


# --- Private Helper Functions: Level 1 (Orchestration) ---

def _prepare_graph_data(
    edges: list[list[int]], vertices: list[int]
) -> tuple[Set[Tuple[int, int]], Set[int], Dict[int, List[int]]]:
    """
    Prepares graph data structures. (Canonical edges, vertex set, adjacency list)
    """
    edges_as_tuples = [tuple(edge) for edge in edges]
    canonical_edges = {tuple(sorted(edge)) for edge in edges_as_tuples}

    if not vertices:
        vertex_set = set(itertools.chain.from_iterable(canonical_edges))
    else:
        vertex_set = set(vertices)

    adj = defaultdict(list)
    for u, v in canonical_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    return canonical_edges, vertex_set, adj


def _build_qubo_for_small_world(
    vertices: Set[int], adj: Dict[int, List[int]], canonical_edges: Set[Tuple[int, int]]
) -> Dict[Tuple[str, str], float]:
    """
    Constructs the QUBO matrix (Q) for the 'small-world' model by orchestrating calls
    to specialized reward-adding functions.
    """
    Q = defaultdict(float)
    all_nodes = list(vertices)
    edge_to_var = {edge: f"x_{edge[0]}_{edge[1]}" for edge in canonical_edges}

    for i, j in itertools.permutations(all_nodes, 2):
        _add_d1_path_reward(Q, i, j, edge_to_var)
        for k in adj.get(i, []):
            if k == i or k == j:
                continue
            if j in adj.get(k, []):
                _add_d2_path_reward(Q, i, j, k, edge_to_var)
    return Q


def _solve_qubo(Q: Dict[Tuple[str, str], float], num_reads: int) -> Dict[str, int]:
    """
    Solves the given QUBO problem using a simulated annealer.
    """
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=num_reads)
    return sampleset.first.sample


def _process_solution(
    best_sample: Dict[str, int], canonical_edges: Set[Tuple[int, int]]
) -> list[EdgeDto]:
    """
    Processes the best sample from the solver into a list of directed EdgeDto objects.
    """
    final_edges = []
    for u_canon, v_canon in canonical_edges:
        var_name = f"x_{u_canon}_{v_canon}"
        if best_sample.get(var_name, 0) == 1:
            final_edges.append(EdgeDto(_from=v_canon, to=u_canon))
        else:
            final_edges.append(EdgeDto(_from=u_canon, to=v_canon))
    return final_edges


# --- Public Service Function ---

def solve_direction_optimization_small_world(
    vertices: list[int], edges: list[list[int]]
) -> list[EdgeDto]:
    """
    Orchestrates the graph direction optimization process using the 'small-world' model.
    """
    if not edges:
        return []

    canonical_edges, vertex_set, adj = _prepare_graph_data(edges, vertices)
    Q = _build_qubo_for_small_world(vertex_set, adj, canonical_edges)
    num_reads = 100
    best_sample = _solve_qubo(Q, num_reads)
    return _process_solution(best_sample, canonical_edges)