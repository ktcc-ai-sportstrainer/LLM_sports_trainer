from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class TechnicalAnalysis(BaseModel):
    swing_mechanics: Dict[str, Any] = Field(..., description="スイングメカニクスの分析")
    identified_issues: List[str] = Field(..., description="特定された課題")
    strengths: List[str] = Field(..., description="強み")
    metrics: Dict[str, float] = Field(..., description="測定された指標")
    recommendations: List[str] = Field(..., description="技術的な改善提案")

class TrainingRecommendation(BaseModel):
    daily_routine: List[Dict[str, Any]] = Field(..., description="日々の練習メニュー")
    key_points: List[str] = Field(..., description="重要なポイント")
    progression_steps: List[str] = Field(..., description="段階的な上達ステップ")
    required_equipment: List[str] = Field(..., description="必要な道具")
    estimated_timeline: str = Field(..., description="予想される習得期間")

class ExecutionMetrics(BaseModel):
    total_time: float = Field(..., description="全体の実行時間（秒）")
    agent_times: Dict[str, float] = Field(..., description="各エージェントの実行時間")
    error_count: int = Field(..., description="発生したエラーの数")
    completion_status: Dict[str, bool] = Field(..., description="各処理の完了状態")

class FinalOutput(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now, description="生成日時")
    technical_analysis: TechnicalAnalysis = Field(..., description="技術分析")
    training_recommendation: TrainingRecommendation = Field(..., description="トレーニング提案")
    execution_metrics: ExecutionMetrics = Field(..., description="実行メトリクス")
    agent_outputs: Dict[str, Any] = Field(..., description="各エージェントの出力履歴")
    summary: str = Field(..., description="全体のまとめ")
    suggestions: List[str] = Field(..., description="今後の提案")
    feedback_points: Optional[List[str]] = Field(default=None, description="フィードバックポイント")