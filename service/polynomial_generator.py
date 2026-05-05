from abc import ABC, abstractmethod

from dimod import BinaryPolynomial

from domain import WeightedGraph

class PolynomialGenerator(ABC):

  @abstractmethod
  def generate_polynomial(self, graph: WeightedGraph) -> BinaryPolynomial:
    pass
