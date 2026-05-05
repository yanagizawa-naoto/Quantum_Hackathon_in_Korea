from dataclasses import dataclass
from domain import WeightedEdge, WeightedGraph

@dataclass
class WeightedEdgeDto:
  vertices: list[int]
  weight: int

  def to_domain(self) -> WeightedEdge:
    return WeightedEdge(self.vertices, self.weight)

@dataclass
class WeightedRequestDto:
  edges: list[WeightedEdgeDto]
  def to_domain(self) -> WeightedGraph:
    return WeightedGraph([edge.to_domain() for edge in self.edges])
