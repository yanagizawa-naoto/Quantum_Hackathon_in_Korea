from dataclasses import dataclass
from typing import List

from service import calculate_total_apsp_distance

@dataclass
class EdgeDto:
    _from: int
    to: int

@dataclass
class ResponseDto:
    edges: List[EdgeDto]
    optimized_graph_score: float
    bidirectional_graph_score: float

    @staticmethod
    def from_tuples(vertices: list[int], tuples: list[tuple[int, int]]) -> "ResponseDto":

      optimized_score = calculate_total_apsp_distance(
        vertices, tuples, is_directed=True
      )

      bidirectional_score = calculate_total_apsp_distance(
        vertices, tuples, is_directed=False
      )

      edges = [EdgeDto(_from=x[0], to=x[1]) for x in tuples]

      return ResponseDto(
        edges=edges,
        optimized_graph_score=optimized_score,
        bidirectional_graph_score=bidirectional_score
      )
