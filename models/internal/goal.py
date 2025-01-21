from pydantic import BaseModel, Field
from typing import List

class Goal(BaseModel):
    primary_goal: str = Field(..., description="主要な目標")
    sub_goals: List[str] = Field(..., description="サブ目標")
    metrics: List[str] = Field(..., description="目標の達成度を測る指標")
