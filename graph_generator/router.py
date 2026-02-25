from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from graph_generator.service import (
    generate_connected_graph,
    compute_planar_faces,
    optimize_edge_orientations,
    save_graph_json,
    list_saved_graphs,
    load_graph_json,
)

router = APIRouter(prefix="/graph", tags=["graph"])


class GraphRequest(BaseModel):
    num_vertices: int = Field(..., ge=1, description="頂点数")
    num_edges: Optional[int] = Field(None, ge=0, description="エッジ数（省略時はランダム）")


class FaceEdge(BaseModel):
    source: int
    target: int


class FacePosition(BaseModel):
    x: float
    y: float


class FacesRequest(BaseModel):
    num_vertices: int = Field(..., ge=1, description="頂点数")
    edges: list[FaceEdge]
    positions: dict[int, FacePosition]
    seed_face_index: Optional[int] = Field(None, description="向きを固定する面のindex")
    seed_orientation: Optional[str] = Field(None, description="cw or ccw")

class SaveGraphRequest(BaseModel):
    name: str = Field(..., min_length=1, description="保存名")
    num_vertices: int = Field(..., ge=1, description="頂点数")
    edges: list[FaceEdge]
    positions: dict[int, FacePosition]


class OptimizeOrientationRequest(BaseModel):
    num_vertices: int = Field(..., ge=1, description="頂点数")
    edges: list[FaceEdge]
    fixed_edges: list[FaceEdge] = Field(default_factory=list, description="固定向きの辺")
    max_iterations: int = Field(1200, ge=100, le=10000, description="最適化反復数")


@router.post("/generate")
async def generate_graph(request: GraphRequest):
    try:
        result = generate_connected_graph(request.num_vertices, request.num_edges)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/faces")
async def list_faces(request: FacesRequest):
    try:
        result = compute_planar_faces(
            request.num_vertices,
            [e.model_dump() for e in request.edges],
            {int(k): v.model_dump() for k, v in request.positions.items()},
            request.seed_face_index,
            request.seed_orientation,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/save")
async def save_graph(request: SaveGraphRequest):
    try:
        data = {
            "num_vertices": request.num_vertices,
            "edges": [e.model_dump() for e in request.edges],
            "positions": {int(k): v.model_dump() for k, v in request.positions.items()},
        }
        result = save_graph_json(request.name, data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/list")
async def list_graphs():
    return {"files": list_saved_graphs()}


@router.get("/load/{name}")
async def load_graph(name: str):
    try:
        data = load_graph_json(name)
        return data
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/optimize-orientation")
async def optimize_orientation(request: OptimizeOrientationRequest):
    try:
        result = optimize_edge_orientations(
            num_vertices=request.num_vertices,
            edges=[e.model_dump() for e in request.edges],
            fixed_edges=[e.model_dump() for e in request.fixed_edges],
            max_iterations=request.max_iterations,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
