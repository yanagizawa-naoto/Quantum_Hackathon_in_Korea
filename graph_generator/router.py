from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from graph_generator.service import generate_connected_graph

router = APIRouter(prefix="/graph", tags=["graph"])


class GraphRequest(BaseModel):
    num_vertices: int = Field(..., ge=1, description="頂点数")
    num_edges: Optional[int] = Field(None, ge=0, description="エッジ数（省略時はランダム）")


@router.post("/generate")
async def generate_graph(request: GraphRequest):
    try:
        result = generate_connected_graph(request.num_vertices, request.num_edges)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
