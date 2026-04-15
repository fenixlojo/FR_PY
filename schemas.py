from pydantic import BaseModel
from typing import List

class CompareResponse(BaseModel):
    matches: List[bool]
    similarities: List[float]
    best_similarity: float
    score: float
    success: bool