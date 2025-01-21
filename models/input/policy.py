from pydantic import BaseModel, Field
from typing import List


class TeachingPolicy(BaseModel):
    focus_points: List[str] = Field(..., description="重点的に指導したい項目")
    teaching_style: str = Field(..., description="指導スタイル")
    goal: str = Field(..., description="達成したい目標")