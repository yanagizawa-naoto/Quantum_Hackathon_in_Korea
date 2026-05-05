from itertools import product

from domain import WeightedGraph
from service.optimization_service import WeightedOptimizationService
from service import calculate_total_apsp_distance


class BruteForceService(WeightedOptimizationService):

  def optimize(self, graph: WeightedGraph) -> list[tuple[int, int]]:
    if graph.is_empty():
      return []

    vertices = graph.get_vertices()
    edges = graph.edges

    best_score = float('inf')
    best_tuples = []

    for bits in product([0, 1], repeat=len(edges)):
      tuples = []
      for i, edge in enumerate(edges):
        if bits[i] == 0:
          tuples.append((edge.vertices[0], edge.vertices[1]))
        else:
          tuples.append((edge.vertices[1], edge.vertices[0]))

      raw_score = calculate_total_apsp_distance(vertices, tuples, is_directed=True)
      score = float('inf') if raw_score == -1 else raw_score
      if score < best_score:
        best_score = score
        best_tuples = tuples

    return best_tuples
