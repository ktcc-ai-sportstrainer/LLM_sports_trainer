from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class BaseballExperience(BaseModel):
    years: int = Field(..., description="野球経験年数")
    history: str = Field(..., description="経験の詳細（所属歴など）")

class DominantHand(BaseModel):
    batting: str = Field(..., description="打撃時の利き手", examples=["右", "左", "両"])
    throwing: str = Field(..., description="投球時の利き手", examples=["右", "左"])

class PhysicalStats(BaseModel):
    height: float = Field(..., description="身長(cm)")
    weight: float = Field(..., description="体重(kg)")
    injuries: Optional[List[str]] = Field(default=None, description="現在または過去の怪我")

class Persona(BaseModel):
    name: str = Field(..., description="選手の名前")
    age: int = Field(..., description="年齢")
    grade: str = Field(..., description="学年（該当する場合）")
    position: str = Field(..., description="ポジション（現在または希望）")
    dominant_hand: DominantHand
    physical_stats: PhysicalStats
    experience: BaseballExperience
    goal: str = Field(..., description="達成したい目標")
    practice_time: str = Field(..., description="普段の練習時間")
    personal_issues: List[str] = Field(..., description="個人的な課題")
    additional_info: Optional[str] = Field(default=None, description="その他の情報")