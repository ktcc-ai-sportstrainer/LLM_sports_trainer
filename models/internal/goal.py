from pydantic import BaseModel, Field
from typing import List, Optional

class Milestone(BaseModel):
    timing: str = Field(..., description="達成予定時期")
    target: str = Field(..., description="達成目標")
    metrics: List[str] = Field(..., description="達成度を測る指標")

class SubGoal(BaseModel):
    description: str = Field(..., description="サブ目標の内容")
    priority: int = Field(..., description="優先度(1-5)")
    timeframe: str = Field(..., description="達成予定期間")
    dependencies: Optional[List[str]] = Field(default=None, description="依存する他の目標")

class Goal(BaseModel):
    primary_goal: str = Field(..., description="主要な目標")
    sub_goals: List[SubGoal] = Field(..., description="サブ目標")
    metrics: List[str] = Field(..., description="目標の達成度を測る指標")
    milestones: List[Milestone] = Field(..., description="段階的な達成目標")
    timeframe: str = Field(..., description="全体の達成予定期間")
    prerequisites: Optional[List[str]] = Field(default=None, description="前提条件")
    constraints: Optional[List[str]] = Field(default=None, description="制約条件")