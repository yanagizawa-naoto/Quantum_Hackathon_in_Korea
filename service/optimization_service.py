from dataclasses import dataclass
from abc import ABC, abstractmethod
from domain import WeightedGraph, WeightedEdge


class WeightedOptimizationService(ABC):

  @abstractmethod
  def optimize(self, graph: WeightedGraph) -> list[tuple[int, int]]:
    pass

class OptimizationService(ABC):

  @abstractmethod
  def optimize(
      self,
      vertices: set[int],
      edges: list[list[int]]
  ) -> list[tuple[int, int]]:
    pass

@dataclass
class ProxyOptimizationService(OptimizationService):

  weighted_optimization_service: WeightedOptimizationService

  def optimize(
      self,
      vertices: set[int],
      edges: list[list[int]]
  ) -> list[tuple[int, int]]:
    edge_list = [WeightedEdge(vertices=edge, weight=1) for edge in edges]
    graph = WeightedGraph(edges=edge_list)
    return self.weighted_optimization_service.optimize(graph)