from pydantic import BaseModel
from typing import List, Any

class AnalyzeResponse(BaseModel):
    text: str
    score: float
    tokens: List[str]
    shap: List[float]
