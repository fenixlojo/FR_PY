from pydantic import BaseModel
from typing import List

class CompareResponse(BaseModel):
    matches: List[bool]
    similarities: List[float]
    score: float
    success: bool