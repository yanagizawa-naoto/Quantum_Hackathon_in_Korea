from dataclasses import dataclass
from typing import Optional


@dataclass
class EstimateResponseDto:
    num_logical_variables: int
    num_quadratic_couplings: int
    num_physical_qubits: int
    max_chain_length: int
    error: Optional[str] = None
