from dimod import BinaryPolynomial, Vartype, SampleSet
from dataclasses import dataclass

from domain import WeightedGraph, WeightedEdge
from service import (
  calculate_total_apsp_distance,
  WeightedOptimizationService,
  PolynomialGenerator,
)
from utils import solve_binary_polynomial, build_bqm, add_polys


@dataclass
class PolynomialOptimizationService(WeightedOptimizationService):

  poly_generators: list[PolynomialGenerator]

  def _build_polynomial(
      self, graph: WeightedGraph
  ) -> BinaryPolynomial:
    terms = BinaryPolynomial({}, Vartype.BINARY)
    for poly_generator in self.poly_generators:
      temp = poly_generator.generate_polynomial(graph)
      terms = add_polys(terms, temp)
    return terms

  def _process_solution(
      self, best_sample: dict[str, int], canonical_edges: list[WeightedEdge]
  ) -> list[tuple[int, int]]:
    """
    Processes the best sample from the solver into a list of directed edge tuples.

    Returns:
        list[tuple[int, int]]: Directed edges represented as (u, v) integer node ID pairs.
    """
    final_edges = []
    for edge in canonical_edges:
      var_name = edge.to_key()
      if best_sample.get(var_name, 0) == 1:
        final_edges.append((edge.vertices[1], edge.vertices[0]))
      else:
        final_edges.append((edge.vertices[0], edge.vertices[1]))
    return final_edges

  def _select_best_sample(
      self,
      sample_set: SampleSet,
      canonical_edges: list[WeightedEdge],
      vertices: set[int]
  ) -> list[tuple[int, int]]:

    def get_effective_score(tuples: list[tuple[int, int]]):
      score = calculate_total_apsp_distance(vertices, tuples, True)
      return float('inf') if score == -1 else score

    return min(
      map(lambda sample: self._process_solution(sample, canonical_edges), sample_set.samples()),
      key=get_effective_score
    )


  # --- Public Service Function ---
  def get_bqm(self, graph: WeightedGraph):
    """
    Builds the BQM (Binary Quadratic Model) for the given graph
    without solving it. Useful for embedding analysis and qubit estimation.
    """
    if graph.is_empty():
      raise ValueError("Cannot build BQM from an empty graph")
    binary_polynomial = self._build_polynomial(graph)
    return build_bqm(binary_polynomial)

  def optimize(
      self,
      graph: WeightedGraph
  ) -> list[tuple[int, int]]:
    """
    Orchestrates the graph direction optimization process by composing the
    configured PolynomialGenerators into a single QUBO polynomial, solving it,
    and selecting the best directed-edge orientation from the resulting samples.
    """
    if graph.is_empty():
      return []

    vertices = graph.get_vertices()
    canonical_edges = graph.edges
    binary_polynomial = self._build_polynomial(graph)

    sample_set = solve_binary_polynomial(binary_polynomial)
    return self._select_best_sample(sample_set, canonical_edges, vertices)