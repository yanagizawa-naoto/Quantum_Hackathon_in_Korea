from typing import List, Optional
from pydantic import BaseModel, Field

class RequestDto(BaseModel):
    vertices: List[int]
    edges: List[List[int]] # Required for /optimize/small-world
    num_edges: Optional[int] = Field(None, ge=0, description="エッジ数（省略時はランダム）")