from pydantic import BaseModel, Field
from typing import List, Optional

class TrainingTask(BaseModel):
    title: str = Field(..., description="トレーニングタスクのタイトル")
    description: str = Field(..., description="詳細な説明")
    duration: str = Field(..., description="予想される所要時間")
    focus_points: List[str] = Field(..., description="注意点")
    equipment: List[str] = Field(..., description="必要な道具")

class TrainingPlan(BaseModel):
    tasks: List[TrainingTask] = Field(..., description="トレーニングタスクのリスト")
    progression_path: str = Field(..., description="上達のためのステップアップの道筋")
    required_time: str = Field(..., description="計画全体の所要時間")