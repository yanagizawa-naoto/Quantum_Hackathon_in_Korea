from dataclasses import dataclass
from typing import List

@dataclass
class RequestDto:
    vertices: List[int]
    edges: List[List[int]]