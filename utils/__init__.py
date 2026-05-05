from importlib import import_module
from typing import Any

__all__ = [
    "extract_vertices",
    "to_canonical_edges",
    "to_adjacency_dict",
    "solve_binary_polynomial",
    "build_bqm",
    "multiply_polys",
    "add_polys",
    "get_indicator_function",
    "estimate_required_qubits",
    "run_with_timeout",
    "TIME_OUT",
]

_MODULE_ATTRS = {
    "extract_vertices": (".graph_utils", "extract_vertices"),
    "to_canonical_edges": (".graph_utils", "to_canonical_edges"),
    "to_adjacency_dict": (".graph_utils", "to_adjacency_dict"),
    "solve_binary_polynomial": (".qubo_utils", "solve_binary_polynomial"),
    "build_bqm": (".qubo_utils", "build_bqm"),
    "multiply_polys": (".qubo_utils", "multiply_polys"),
    "add_polys": (".qubo_utils", "add_polys"),
    "get_indicator_function": (".qubo_utils", "get_indicator_function"),
    "estimate_required_qubits": (".embedding_utils", "estimate_required_qubits"),
    "run_with_timeout": (".timeout", "run_with_timeout"),
    "TIME_OUT": (".timeout", "TIME_OUT"),
}


def __getattr__(name: str) -> Any:
    """
    Lazily import attributes from submodules when accessed.

    This avoids eagerly importing heavy dependencies when `utils`
    is imported, while preserving the public API.
    """
    try:
        module_name, attr_name = _MODULE_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
