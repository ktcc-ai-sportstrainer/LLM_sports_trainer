from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np

class Joint(BaseModel):
    x: float = Field(..., description="X座標")
    y: float = Field(..., description="Y座標")
    z: float = Field(..., description="Z座標")
    confidence: Optional[float] = Field(default=None, description="検出信頼度")

class Frame(BaseModel):
    joints: List[Joint] = Field(..., description="フレーム内の関節位置")
    timestamp: float = Field(..., description="フレームのタイムスタンプ")

class SwingPhase(BaseModel):
    name: str = Field(..., description="フェーズ名")
    start_frame: int = Field(..., description="開始フレーム")
    end_frame: int = Field(..., description="終了フレーム")
    key_points: List[str] = Field(..., description="重要なポイント")

class SwingMetrics(BaseModel):
    bat_speed: float = Field(..., description="バットスピード(m/s)")
    rotation_sequence: float = Field(..., description="回転の連動性スコア(0-1)")
    weight_shift: float = Field(..., description="重心移動の効率性スコア(0-1)")
    hip_shoulder_separation: float = Field(..., description="腰肩の分離度(度)")
    contact_accuracy: Optional[float] = Field(default=None, description="ミート精度スコア(0-1)")

class SwingAnalysis(BaseModel):
    phases: List[SwingPhase] = Field(..., description="スイングの各フェーズ")
    metrics: SwingMetrics = Field(..., description="計測された指標")
    issues_found: List[str] = Field(..., description="検出された技術的課題")
    strengths: List[str] = Field(..., description="良い点")
    recommendations: List[str] = Field(..., description="改善のための提案")

class MotionData(BaseModel):
    frames: List[Frame] = Field(..., description="全フレームのデータ")
    fps: int = Field(..., description="フレームレート")
    total_frames: int = Field(..., description="総フレーム数")
    analysis: SwingAnalysis = Field(..., description="動作解析結果")
    
    class Config:
        arbitrary_types_allowed = True  # numpyアレイのサポート用