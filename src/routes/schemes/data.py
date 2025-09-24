from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    file_id: str = None
    chunk_size: Optional[int] = 100
    overlap_size: Optional[int] = 20
    do_reset: Optional[int] = 0


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    include_sources: Optional[bool] = True
    similarity_threshold: Optional[float] = 0.7    
