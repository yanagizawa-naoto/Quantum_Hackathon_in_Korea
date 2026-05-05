from fastapi import APIRouter, HTTPException

from dto import WeightedRequestDto, ResponseDto, EstimateResponseDto
from service import (
  FlowConservationPolynomialGenerator,
  MinimizeSumOfApspPolynomialGenerator,
  SmallWorldSpec,
  NHop,
  PolynomialOptimizationService,
  NaotoService,
  BruteForceService,
)
from utils import run_with_timeout, estimate_required_qubits

router = APIRouter()

small_world_service = PolynomialOptimizationService(
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

naoto_service = NaotoService()

brute_force_service = BruteForceService()

def _run_optimization(service, graph):
  tuples = service.optimize(graph)
  return ResponseDto.from_tuples(list(graph.get_vertices()), tuples)

@router.post("/api/v1/mr2s", response_model=ResponseDto)
async def optimize_by_small_world(request: WeightedRequestDto):
  try:
    graph = request.to_domain()
    return await run_with_timeout(_run_optimization, small_world_service, graph)
  except HTTPException:
    raise
  except ValueError as e:
    raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

@router.post("/api/v1/raw-sa", response_model=ResponseDto)
async def optimize_by_naoto(request: WeightedRequestDto):
  try:
    graph = request.to_domain()
    return await run_with_timeout(_run_optimization, naoto_service, graph)
  except HTTPException:
    raise
  except ValueError as e:
    raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

@router.post("/api/v1/brute-force", response_model=ResponseDto)
async def optimize_brute_force(request: WeightedRequestDto):
  try:
    graph = request.to_domain()
    return await run_with_timeout(_run_optimization, brute_force_service, graph)
  except HTTPException:
    raise
  except ValueError as e:
    raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

@router.post("/api/v1/mr2s/estimate", response_model=EstimateResponseDto)
async def estimate_qubits(request: WeightedRequestDto):
  try:
    graph = request.to_domain()
    bqm = small_world_service.get_bqm(graph)
    result = estimate_required_qubits(bqm)
    return EstimateResponseDto(**result)
  except ValueError as e:
    raise HTTPException(status_code=400, detail=f"Invalid input: {e}")
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Estimation failed: {e}")
