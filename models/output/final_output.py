from pydantic import BaseModel, Field
from typing import List, Dict, Any

class TechnicalAnalysis(BaseModel):
    swing_mechanics: Dict[str, Any] = Field(..., description="スイングメカニクスの分析")
    identified_issues: List[str] = Field(..., description="特定された課題")
    strengths: List[str] = Field(..., description="強み")

class TrainingRecommendation(BaseModel):
    daily_routine: List[Dict[str, Any]] = Field(..., description="日々の練習メニュー")
    key_points: List[str] = Field(..., description="重要なポイント")
    progression_steps: List[str] = Field(..., description="段階的な上達ステップ")

class FinalOutput(BaseModel):
    technical_analysis: TechnicalAnalysis = Field(..., description="技術分析")
    training_recommendation: TrainingRecommendation = Field(..., description="トレーニング提案")
    agent_outputs: List[AgentOutput] = Field(..., description="各エージェントの出力履歴")
    summary: str = Field(..., description="総括")