import itertools
from collections import defaultdict

def to_adjacency_dict(canonical_edges: set[tuple[int, int]]):
  adj = defaultdict(list)
  for u, v in canonical_edges:
    adj[u].append(v)
    adj[v].append(u)
  return adj

def extract_vertices(edges: list[list[int]], vertices: list[int]) -> set[int]:
  if not vertices:
    vertex_set = set(itertools.chain.from_iterable(edges))
  else:
    vertex_set = set(vertices)
  return vertex_set

def to_canonical_edges(edges: list[list[int]]) -> set[tuple[int, int]]:
  edges_as_tuples = [tuple(edge) for edge in edges]
  canonical_edges = {tuple(sorted(edge)) for edge in edges_as_tuples}
  return canonical_edges