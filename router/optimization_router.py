from fastapi import APIRouter, HTTPException
import itertools

from dto import RequestDto, ResponseDto, EdgeDto
from service import (
    calculate_total_apsp_distance,
    optimize_edge_orientations,
    FlowConservationPolynomialGenerator,
    MinimizeSumOfApspPolynomialGenerator,
    SmallWorldSpec,
    NHop,
    ProxyOptimizationService,
    PolynomialOptimizationService,
)
from utils import extract_vertices

router = APIRouter()

small_world_service = ProxyOptimizationService(
  PolynomialOptimizationService(
    [
      FlowConservationPolynomialGenerator(),
      MinimizeSumOfApspPolynomialGenerator(
        SmallWorldSpec(
          n_hops=[
            NHop(n=2, weight=1),
            NHop(n=3, weight=1)
          ]
        )
      )
    ]
  )
)

@router.post("/optimize/small-world", response_model=ResponseDto)
async def optimize_graph_direction(request: RequestDto):
  """
  API endpoint to optimize graph edge directions.

  This endpoint orchestrates the optimization and the scoring calculation.
  1. Calls the optimization service to get the directed graph.
  2. Calls the graph analyzer to calculate the APSP score for the new graph.
  3. Calls the graph analyzer to calculate the APSP score for the original
     bidirectional graph for comparison.
  4. Returns the final response including the graph and scores.
  """
  try:
    vertices_set = extract_vertices(request.edges, request.vertices)
    tuples = small_world_service.optimize(vertices_set, request.edges)
    return ResponseDto.from_tuples(request.vertices, tuples)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

@router.post("/optimize/naoto", response_model=ResponseDto)
async def optimize_graph_direction_goodgood_meathod(request: RequestDto):
    """
    API endpoint to optimize graph edge directions using the 'goodgood' method.
    This method uses the `optimize_edge_orientations` service.
    """
    try:
        if request.vertices:
            vertex_set = set(request.vertices)
            num_vertices = len(vertex_set)
        else:
            vertex_set = set(itertools.chain.from_iterable(request.edges))
            num_vertices = len(vertex_set) if vertex_set else 0

        if num_vertices == 0:
            return ResponseDto(edges=[], optimized_graph_score=0, bidirectional_graph_score=0)

        edges_for_service = [{"source": u, "target": v} for u, v in request.edges]

        optimization_result = optimize_edge_orientations(
            num_vertices=num_vertices,
            edges=edges_for_service
        )

        optimized_edges_dto = [
            EdgeDto(_from=edge['source'], to=edge['target'])
            for edge in optimization_result['directed_edges']
        ]

        optimized_edges_tuples = [(e._from, e.to) for e in optimized_edges_dto]
        optimized_score = calculate_total_apsp_distance(
          vertex_set,
          optimized_edges_tuples,
          is_directed=True
        )

        original_edges_tuples = [tuple(edge) for edge in request.edges]
        bidirectional_score = calculate_total_apsp_distance(
          vertex_set,
          original_edges_tuples,
          is_directed=False
        )

        return ResponseDto(
            edges=optimized_edges_dto,
            optimized_graph_score=optimized_score,
            bidirectional_graph_score=bidirectional_score
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")
