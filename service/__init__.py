from importlib import import_module
from typing import Any

__all__ = [
    "OptimizationService",
    "WeightedOptimizationService",
    "ProxyOptimizationService",
    "FlowConservationPolynomialGenerator",
    "MinimizeSumOfApspPolynomialGenerator",
    "PolynomialGenerator",
    "PolynomialOptimizationService",
    "SmallWorldSpec",
    "NHop",
    "calculate_total_apsp_distance",
    "BruteForceService",
    "NaotoService",
    "optimize_edge_orientations",
    "generate_connected_graph",
    "compute_planar_faces",
]

_MODULE_ATTRS = {
    "OptimizationService": (".optimization_service", "OptimizationService"),
    "WeightedOptimizationService": (".optimization_service", "WeightedOptimizationService"),
    "ProxyOptimizationService": (".optimization_service", "ProxyOptimizationService"),
    "FlowConservationPolynomialGenerator": (
        ".flow_conservation_polynomial_generator",
        "FlowConservationPolynomialGenerator",
    ),
    "MinimizeSumOfApspPolynomialGenerator": (
        ".minimize_sum_of_apsp_polynomial_generator",
        "MinimizeSumOfApspPolynomialGenerator",
    ),
    "PolynomialGenerator": (".polynomial_generator", "PolynomialGenerator"),
    "PolynomialOptimizationService": (
        ".polynomial_optimization_service",
        "PolynomialOptimizationService",
    ),
    "SmallWorldSpec": (".minimize_sum_of_apsp_polynomial_generator", "SmallWorldSpec"),
    "NHop": (".minimize_sum_of_apsp_polynomial_generator", "NHop"),
    "calculate_total_apsp_distance": (".graph_analyzer", "calculate_total_apsp_distance"),
    "BruteForceService": (".bruteforce_service", "BruteForceService"),
    "NaotoService": (".naoto_service", "NaotoService"),
    "optimize_edge_orientations": (".naoto_service", "optimize_edge_orientations"),
    "generate_connected_graph": (".naoto_service", "generate_connected_graph"),
    "compute_planar_faces": (".naoto_service", "compute_planar_faces"),
}


def __getattr__(name: str) -> Any:
    """
    Lazily import attributes from submodules when accessed.

    This avoids eagerly importing heavy dependencies when `service`
    is imported, while preserving the public API.
    """
    try:
        module_name, attr_name = _MODULE_ATTRS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
