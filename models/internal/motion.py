from pydantic import BaseModel, Field
from typing import List, Dict
import numpy as np


class MotionData(BaseModel):
    keypoints_3d: List[Dict[str, List[float]]] = Field(..., description="3D姿勢データ")
    frame_count: int = Field(..., description="フレーム数")
    
    class Config:
        arbitrary_types_allowed = True  # numpy arrayのサポート用

class SwingAnalysis(BaseModel):
    phase_timings: Dict[str, int] = Field(..., description="スイングの各フェーズのタイミング")
    key_metrics: Dict[str, float] = Field(..., description="重要な指標（角度、速度など）")
    issues_found: List[str] = Field(..., description="発見された技術的な課題")
    strengths: List[str] = Field(..., description="良い点")