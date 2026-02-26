from fastapi import APIRouter, HTTPException

from dto.RequestDto import RequestDto
from dto.ResponseDto import ResponseDto
from service.optimization_service_small_world import solve_direction_optimization_small_world
from service.graph_analyzer import calculate_total_apsp_distance
import itertools

router = APIRouter()

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
    # 1. Get the optimized directed graph
    optimized_edges_dto = solve_direction_optimization_small_world(request.vertices, request.edges)

    # Prepare data for scoring
    if request.vertices:
      vertex_set = set(request.vertices)
    else:
      vertex_set = set(itertools.chain.from_iterable(request.edges))

    # 2. Calculate score for the optimized directed graph
    optimized_edges_tuples = [(e._from, e.to) for e in optimized_edges_dto]
    optimized_score = calculate_total_apsp_distance(
      vertex_set, optimized_edges_tuples, is_directed=True
    )

    # 3. Calculate score for the original bidirectional graph
    # Convert original List[List[int]] to List[Tuple[int, int]] for the analyzer
    original_edges_tuples = [tuple(edge) for edge in request.edges]
    bidirectional_score = calculate_total_apsp_distance(
      vertex_set, original_edges_tuples, is_directed=False
    )

    # 4. Construct and return the final response
    return ResponseDto(
      edges=optimized_edges_dto,
      optimized_graph_score=optimized_score,
      bidirectional_graph_score=bidirectional_score
    )
  except ValueError as e:
    raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")
