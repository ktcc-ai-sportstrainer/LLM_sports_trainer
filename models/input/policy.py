from pydantic import BaseModel, Field
from typing import List, Optional

class TeachingPolicy(BaseModel):
    philosophy: str = Field(..., description="指導の基本方針")
    focus_points: List[str] = Field(..., description="重点的に指導したい項目")
    teaching_style: str = Field(..., description="指導スタイル", 
                               examples=["詳細な技術指導", "メンタル面重視", "基礎重視", "実践重視"])
    short_term_goals: List[str] = Field(..., description="短期的な目標")
    long_term_goals: List[str] = Field(..., description="長期的な目標")
    player_strengths: List[str] = Field(..., description="指導者から見た選手の強み")
    player_weaknesses: List[str] = Field(..., description="指導者から見た選手の弱み")
    training_constraints: Optional[List[str]] = Field(default=None, description="練習における制約条件")
    additional_notes: Optional[str] = Field(default=None, description="その他の注意点や指導方針")