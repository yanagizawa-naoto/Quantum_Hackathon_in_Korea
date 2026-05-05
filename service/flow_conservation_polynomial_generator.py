from dimod import BinaryPolynomial, Vartype

from domain import WeightedGraph, AdjEntry
from service import PolynomialGenerator
from utils import get_indicator_function, add_polys, multiply_polys


class FlowConservationPolynomialGenerator(PolynomialGenerator):

  def _get_a_term(self, vertex: int, adj_vertices: list[AdjEntry]) -> BinaryPolynomial:
    term = BinaryPolynomial({}, Vartype.BINARY)
    for adj_vertex in adj_vertices:
      indicator_function = get_indicator_function(vertex, adj_vertex.vertex, adj_vertex.weight)
      indicator_function.scale(2)
      temp = add_polys(indicator_function, BinaryPolynomial({(): -1}, Vartype.BINARY))
      term = add_polys(term, temp)
    return multiply_polys(term, term)

  def _get_total_term(self, vertices: set[int], adj: dict[int, list[AdjEntry]]) -> BinaryPolynomial:
    term = BinaryPolynomial({}, Vartype.BINARY)

    for vertex in vertices:
      temp = self._get_a_term(vertex, adj.get(vertex, []))
      term = add_polys(term, temp)

    return term

  def generate_polynomial(self, graph: WeightedGraph) -> BinaryPolynomial:
    if graph.is_empty():
      return BinaryPolynomial({}, Vartype.BINARY)

    vertices = graph.get_vertices()
    adj = graph.get_adjacency_dict()

    return self._get_total_term(vertices, adj)