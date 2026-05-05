from dimod import BinaryQuadraticModel
from minorminer import find_embedding
import dwave_networkx as dnx

_target_graph = dnx.pegasus_graph(16)


def estimate_required_qubits(bqm: BinaryQuadraticModel) -> dict:
    """
    Estimates the number of physical qubits required to embed the given BQM
    onto a Pegasus P16 quantum annealer topology using minorminer.

    Returns a dict with:
      - num_logical_variables: number of variables in the BQM
      - num_quadratic_couplings: number of quadratic interactions
      - num_physical_qubits: total physical qubits after embedding
      - max_chain_length: longest chain in the embedding
    """
    num_logical = len(bqm.variables)
    num_couplings = bqm.num_interactions

    if num_logical == 0:
        return {
            "num_logical_variables": 0,
            "num_quadratic_couplings": 0,
            "num_physical_qubits": 0,
            "max_chain_length": 0,
        }

    source_edgelist = list(bqm.quadratic)

    embedding = find_embedding(source_edgelist, _target_graph)

    if not embedding:
        return {
            "num_logical_variables": num_logical,
            "num_quadratic_couplings": num_couplings,
            "num_physical_qubits": -1,
            "max_chain_length": -1,
            "error": "Embedding not found for this problem size on Pegasus P16",
        }

    total_physical_qubits = sum(len(chain) for chain in embedding.values())
    max_chain_length = max(len(chain) for chain in embedding.values())

    return {
        "num_logical_variables": num_logical,
        "num_quadratic_couplings": num_couplings,
        "num_physical_qubits": total_physical_qubits,
        "max_chain_length": max_chain_length,
    }
