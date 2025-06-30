from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class Equipment(BaseModel):
    name: str = Field(..., description="道具の名前")
    quantity: int = Field(default=1, description="必要な数量")
    alternative: Optional[str] = Field(default=None, description="代替可能な道具")

class TrainingTask(BaseModel):
    title: str = Field(..., description="トレーニングタスクのタイトル")
    description: str = Field(..., description="詳細な説明")
    duration: str = Field(..., description="予想される所要時間")
    focus_points: List[str] = Field(..., description="注意点")
    equipment: List[Equipment] = Field(..., description="必要な道具")
    difficulty: int = Field(..., description="難易度(1-5)")
    prerequisites: Optional[List[str]] = Field(default=None, description="前提条件")
    variations: Optional[List[str]] = Field(default=None, description="バリエーション")

class ProgressionStep(BaseModel):
    phase: str = Field(..., description="段階名")
    duration: str = Field(..., description="予想期間")
    tasks: List[str] = Field(..., description="含まれるタスクのID")
    success_criteria: List[str] = Field(..., description="達成基準")

class Schedule(BaseModel):
    day: str = Field(..., description="曜日または日付")
    tasks: List[str] = Field(..., description="タスクのID")
    total_duration: str = Field(..., description="合計時間")
    notes: Optional[str] = Field(default=None, description="特記事項")

class TrainingPlan(BaseModel):
    tasks: List[TrainingTask] = Field(..., description="トレーニングタスクのリスト")
    progression_path: List[ProgressionStep] = Field(..., description="上達のためのステップアップの道筋")
    weekly_schedule: Dict[str, Schedule] = Field(..., description="週間スケジュール")
    required_time: str = Field(..., description="計画全体の所要時間")
    rest_days: List[str] = Field(..., description="推奨休養日")
    evaluation_points: List[str] = Field(..., description="進捗評価ポイント")
    adjustments: Optional[List[Dict[str, str]]] = Field(default=None, description="状況に応じた調整方法")