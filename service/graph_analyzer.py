import networkx as nx
from typing import List, Tuple, Set

def calculate_total_apsp_distance(
    vertices: Set[int], edges: List[Tuple[int, int]], is_directed: bool
) -> float:
    """
    Calculates the sum of all-pairs shortest path (APSP) lengths for a given graph.

    Args:
        vertices: A set of all vertex IDs in the graph.
        edges: A list of edges, where each edge is a tuple (u, v).
        is_directed: A boolean indicating whether the graph is directed.

    Returns:
        The sum of all shortest path lengths. If the graph is not strongly connected
        (for directed) or not connected (for undirected), unreachable pairs are
        not included in the sum.
    """
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    G.add_nodes_from(vertices)
    G.add_edges_from(edges)

    total_distance = 0
    
    # networkx.all_pairs_shortest_path_length returns an iterator of (source, {target: length})
    path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    # Sum up all the path lengths
    for source in vertices:
        if source not in path_lengths:
            continue
        for target in vertices:
            if source == target:
                continue
            
            # If a target is not reachable from source, it won't be in the path_lengths dict.
            # This is the desired behavior - we only sum over existing paths.
            distance = path_lengths[source].get(target)
            if distance is not None:
                total_distance += distance

    return total_distance
