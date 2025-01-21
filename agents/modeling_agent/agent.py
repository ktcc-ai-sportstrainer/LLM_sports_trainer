import numpy as np
import torch
import json
import os
from typing import Dict, Any, List, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base import BaseAgent
from utils.video import VideoProcessor
from models.internal.motion import MotionData, SwingAnalysis

class ModelingAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        self.video_processor = VideoProcessor()
        
        # プロンプトの読み込み
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
            
        self.analysis_prompt = ChatPromptTemplate.from_template(prompts["analysis_prompt"])
        self.metrics = SwingMetrics()  # スイング分析用メトリクスクラス
        
    async def run(self, video_path: str) -> Dict[str, Any]:
        """動画を受け取り、3D姿勢推定と動作解析を行う"""
        # 1. 動画からフレームを抽出
        frames = self.video_processor.read_video(video_path)
        
        # 2. 3D姿勢推定の実行
        keypoints_3d = await self._estimate_3d_poses(frames)
        
        # 3. スイングフェーズの検出
        phases = self.metrics.detect_swing_phases(keypoints_3d)
        
        # 4. 各種メトリクスの計算
        metrics = await self._calculate_metrics(keypoints_3d, phases)
        
        # 5. 技術的な分析の生成
        analysis_result = await self._generate_analysis(phases, metrics)
        
        # 6. 可視化データの生成
        visualization_data = self._generate_visualization(keypoints_3d, phases)
        
        return self.create_output(
            output_type="motion_analysis",
            content={
                "motion_data": MotionData(
                    keypoints_3d=keypoints_3d.tolist(),
                    frame_count=len(frames)
                ).dict(),
                "swing_analysis": SwingAnalysis(
                    phase_timings=phases,
                    key_metrics=metrics,
                    issues_found=analysis_result["issues"],
                    strengths=analysis_result["strengths"]
                ).dict(),
                "visualization": visualization_data
            }
        ).dict()

    async def _estimate_3d_poses(self, frames: np.ndarray) -> np.ndarray:
        """MotionAGFormerによる3D姿勢推定"""
        model = self._load_model()
        
        # バッチ処理のための準備
        processed_frames = self._preprocess_frames(frames)
        
        # 推定の実行
        with torch.no_grad():
            predictions = model(processed_frames)
        
        # 後処理
        keypoints_3d = self._postprocess_predictions(predictions)
        
        return keypoints_3d

    async def _calculate_metrics(
        self,
        keypoints_3d: np.ndarray,
        phases: Dict[str, int]
    ) -> Dict[str, float]:
        """スイング動作の各種メトリクスを計算"""
        metrics = {}
        
        # バットスピード
        metrics["bat_speed"] = self.metrics.calculate_bat_speed(
            keypoints_3d, phases["contact"]
        )
        
        # 回転速度（腰、肩）
        hip_speed = self.metrics.calculate_rotation_speed(keypoints_3d, "hips")
        shoulder_speed = self.metrics.calculate_rotation_speed(keypoints_3d, "shoulders")
        metrics["hip_rotation_speed"] = hip_speed
        metrics["shoulder_rotation_speed"] = shoulder_speed
        
        # 回転の連動性
        metrics["rotation_sequence"] = self.metrics.evaluate_rotation_sequence(
            keypoints_3d, phases
        )
        
        # 重心移動
        metrics["weight_shift"] = self.metrics.analyze_weight_shift(keypoints_3d, phases)
        
        # スイング平面
        metrics["swing_plane_angle"] = self.metrics.calculate_swing_plane(keypoints_3d)
        
        return metrics

    async def _generate_analysis(
        self,
        phases: Dict[str, int],
        metrics: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """メトリクスを基に技術的な分析を生成"""
        response = await self.llm.ainvoke(
            self.analysis_prompt.format(
                phases=json.dumps(phases, ensure_ascii=False),
                metrics=json.dumps(metrics, ensure_ascii=False)
            )
        )
        
        # 応答をパース
        analysis = json.loads(response.content)
        return {
            "issues": analysis["issues"],
            "strengths": analysis["strengths"]
        }

    def _generate_visualization(
        self,
        keypoints_3d: np.ndarray,
        phases: Dict[str, int]
    ) -> Dict[str, Any]:
        """可視化用のデータを生成"""
        # キーポイントの軌跡データ
        trajectories = self._calculate_trajectories(keypoints_3d)
        
        # フェーズごとのハイライト
        highlights = self._create_phase_highlights(keypoints_3d, phases)
        
        return {
            "trajectories": trajectories,
            "phase_highlights": highlights,
            "keyframes": self._select_keyframes(keypoints_3d, phases)
        }

    def _load_model(self):
        """MotionAGFormerモデルの読み込み"""
        model = MotionAGFormer(
            n_layers=16,
            dim_feat=128,
            dim_rep=512,
            dim_out=3
        )
        
        checkpoint = torch.load('checkpoint/motionagformer-b-h36m.pth.tr')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        return model.cuda() if torch.cuda.is_available() else model

    # その他の補助メソッド（_preprocess_frames, _postprocess_predictions等）は省略