from dataclasses import dataclass
from typing import List

@dataclass
class EdgeDto:
    _from: int
    to: int

@dataclass
class ResponseDto:
    edges: List[EdgeDto]