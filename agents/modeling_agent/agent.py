import numpy as np
import torch
from typing import Dict, Any, List
import json

class ModelingAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        self.video_processor = VideoProcessor()
        
    async def run(self, video_path: str) -> Dict[str, Any]:
        """
        動画を受け取り、3D姿勢推定と動作解析を行う
        """
        # 1. 動画からフレームを抽出
        frames, width, height = self.video_processor.read_video(video_path)
        
        # 2. MotionAGFormerによる3D姿勢推定
        motion_data = await self._estimate_3d_poses(frames)
        keypoints_3d = motion_data["keypoints_3d"]
        
        # 3. スイング動作の解析
        swing_analysis = await self._analyze_swing(keypoints_3d)
        
        # 4. 言語的な解析結果の生成
        analysis_description = await self._generate_analysis_description(swing_analysis)
        
        return self.create_output(
            output_type="motion_analysis",
            content={
                "motion_data": motion_data,
                "swing_analysis": swing_analysis,
                "analysis_description": analysis_description
            }
        ).dict()

    async def _estimate_3d_poses(self, frames: np.ndarray) -> Dict[str, Any]:
        """MotionAGFormerを使用して3D姿勢推定を実行"""
        # MotionAGFormerのセットアップ
        args = {}  # MotionAGFormerの設定
        args.update({
            'n_layers': 16,
            'dim_feat': 128,
            'dim_rep': 512,
            'dim_out': 3
        })
        model = MotionAGFormer(**args)
        
        # モデルの重みを読み込み
        checkpoint = torch.load('checkpoint/motionagformer-b-h36m.pth.tr')
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        # フレームの前処理とバッチ処理
        processed_frames = self._preprocess_frames(frames)
        with torch.no_grad():
            keypoints_3d = model(processed_frames)
        
        # 後処理（必要に応じて座標系の変換など）
        keypoints_3d = self._postprocess_keypoints(keypoints_3d)
        
        return {
            "keypoints_3d": keypoints_3d,
            "frame_count": len(frames)
        }

    async def _analyze_swing(self, keypoints_3d: np.ndarray) -> Dict[str, Any]:
        """スイング動作の解析を行う"""
        # 1. フェーズの特定
        phases = self._identify_swing_phases(keypoints_3d)
        
        # 2. 重要な指標の計算
        metrics = self._calculate_metrics(keypoints_3d, phases)
        
        # 3. 技術的な課題と強みの特定
        issues, strengths = await self._identify_issues_and_strengths(
            keypoints_3d, phases, metrics
        )
        
        return {
            "phase_timings": phases,
            "key_metrics": metrics,
            "issues_found": issues,
            "strengths": strengths
        }

    def _identify_swing_phases(self, keypoints_3d: np.ndarray) -> Dict[str, int]:
        """スイングの各フェーズのタイミングを特定"""
        # キーポイントの動きからフェーズを検出
        phases = {
            "stance": 0,
            "load": 0,
            "stride": 0,
            "contact": 0,
            "follow_through": 0
        }
        
        # バットの動きと体の回転から各フェーズのフレームを特定
        bat_movement = self._calculate_bat_movement(keypoints_3d)
        body_rotation = self._calculate_body_rotation(keypoints_3d)
        
        # フェーズの境界を検出
        phases["load"] = self._detect_load_phase(bat_movement, body_rotation)
        phases["stride"] = self._detect_stride_phase(keypoints_3d)
        phases["contact"] = self._detect_contact_phase(bat_movement)
        phases["follow_through"] = self._detect_follow_through(bat_movement)
        
        return phases

    def _calculate_metrics(self, keypoints_3d: np.ndarray, phases: Dict[str, int]) -> Dict[str, float]:
        """重要な指標を計算"""
        metrics = {}
        
        # バットスピード
        metrics["bat_speed"] = self._calculate_bat_speed(keypoints_3d, phases["contact"])
        
        # 体の回転速度
        metrics["hip_rotation_speed"] = self._calculate_rotation_speed(keypoints_3d, "hips")
        metrics["shoulder_rotation_speed"] = self._calculate_rotation_speed(keypoints_3d, "shoulders")
        
        # 重心移動
        metrics["weight_shift"] = self._analyze_weight_shift(keypoints_3d, phases)
        
        # スイング平面
        metrics["swing_plane_angle"] = self._calculate_swing_plane(keypoints_3d)
        
        return metrics

    async def _identify_issues_and_strengths(
        self,
        keypoints_3d: np.ndarray,
        phases: Dict[str, int],
        metrics: Dict[str, float]
    ) -> tuple[List[str], List[str]]:
        """技術的な課題と強みを特定"""
        issues = []
        strengths = []
        
        # メトリクスを基準値と比較
        if metrics["bat_speed"] < 70:  # 仮の基準値
            issues.append("バットスピードが遅い")
        else:
            strengths.append("十分なバットスピードがある")
            
        if metrics["hip_rotation_speed"] < metrics["shoulder_rotation_speed"]:
            issues.append("下半身から順番に回転できていない")
        else:
            strengths.append("下半身主導の良い回転順序")
        
        # 他の分析項目も同様に評価
        
        return issues, strengths

    async def _generate_analysis_description(self, analysis: Dict[str, Any]) -> str:
        """分析結果を自然言語で説明"""
        prompt = ChatPromptTemplate.from_template(
            """以下のスイング分析結果を、コーチが選手に説明するような自然な言葉で表現してください。
            
            フェーズタイミング:
            {phase_timings}
            
            測定された指標:
            {metrics}
            
            発見された課題:
            {issues}
            
            強み:
            {strengths}
            
            以下の点に注意して説明を生成してください：
            1. 技術的な観点から重要な点を優先的に説明
            2. 改善点は建設的な表現を使用
            3. 良い点もしっかりと言及
            4. 具体的な数値は必要な場合のみ言及
            """
        )
        
        response = await self.llm.ainvoke(
            prompt.format(
                phase_timings=json.dumps(analysis["phase_timings"], ensure_ascii=False),
                metrics=json.dumps(analysis["key_metrics"], ensure_ascii=False),
                issues="\n".join(analysis["issues_found"]),
                strengths="\n".join(analysis["strengths"])
            )
        )
        
        return response.content