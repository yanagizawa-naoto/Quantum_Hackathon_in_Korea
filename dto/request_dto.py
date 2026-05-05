from typing import Optional
from pydantic import BaseModel, Field

class RequestDto(BaseModel):
    vertices: list[int]
    edges: list[list[int]] # Required for /optimize/small-world
    num_edges: Optional[int] = Field(None, ge=0, description="エッジ数（省略時はランダム）")