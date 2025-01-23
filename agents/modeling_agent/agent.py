from typing import Any, Dict, List
import json
import os
import sys
import subprocess
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base import BaseAgent
from utils.video import VideoProcessor

class ModelingAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI):
        """
        llm: ChatOpenAI のインスタンス
        """
        super().__init__(llm)
        self.video_processor = VideoProcessor()

        # プロンプトの読み込み
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        self.analysis_prompt = ChatPromptTemplate.from_template(prompts["analysis_prompt"])
        self.visualization_prompt = ChatPromptTemplate.from_template(prompts["visualization_prompt"])
        self.comparison_prompt = ChatPromptTemplate.from_template(prompts["comparison_prompt"])

    async def run(self, user_video_path: str, ideal_video_path: str) -> Dict[str, Any]:
        """
        2つの動画（ユーザーのスイングと理想のスイング）を
        MotionAGFormer + JsonAnalist で解析し、差分比較。
        """
        try:
            # 1. 3D推定
            user_json_path = await self._estimate_3d_poses(user_video_path, "output/user_swing")
            ideal_json_path = await self._estimate_3d_poses(ideal_video_path, "output/ideal_swing")

            # 2. 分析
            user_analysis = await self._analyze_3d_json(user_json_path)
            ideal_analysis = await self._analyze_3d_json(ideal_json_path)

            # 3. 差分
            differences = self._calculate_differences(user_analysis, ideal_analysis)

            # 4. LLMに比較用データを投げてフィードバックを作る
            comparison_feedback = await self._generate_comparison_feedback(
                user_analysis, ideal_analysis, differences
            )

            return {
                "user_analysis": user_analysis,
                "ideal_analysis": ideal_analysis,
                "differences": differences,
                "comparison_feedback": comparison_feedback
            }

        except Exception as e:
            self.logger.log_error(f"Error in ModelingAgent: {str(e)}")
            return {}

    async def _estimate_3d_poses(self, video_path: str, output_dir: str) -> str:
        """MotionAGFormer/run/vis.py を呼び出して 3D姿勢推定json を生成"""
        os.makedirs(output_dir, exist_ok=True)

        # 絶対パスを使用
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vis_script = os.path.join(base_dir, "MotionAGFormer", "run", "vis.py")
        video_path = os.path.abspath(video_path)
        output_dir = os.path.abspath(output_dir)

        if not os.path.exists(vis_script):
            raise FileNotFoundError(f"vis.py script not found at {vis_script}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")

        cmd = [
            sys.executable,
            vis_script,
            "--video", video_path,
            "--output", output_dir
        ]
        subprocess.run(cmd, check=True)

        # 出力されるJSONファイルのパス
        json_path = os.path.join(output_dir, "3d_result.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"3D pose estimation failed, JSON not found at {json_path}")

        return json_path

    async def _analyze_3d_json(self, json_file_path: str) -> Dict[str, Any]:
        """JsonAnalist.py で解析し、stdoutに出る結果を JSON ロードして返す"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        analyzer_script = os.path.join(base_dir, "MotionAGFormer", "JsonAnalist.py")

        if not os.path.exists(analyzer_script):
            raise FileNotFoundError(f"JsonAnalist.py not found at {analyzer_script}")

        cmd = [sys.executable, analyzer_script, "--input", json_file_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

        try:
            analysis_result = json.loads(proc.stdout)
            return analysis_result
        except json.JSONDecodeError as e:
            self.logger.log_error(f"Error parsing analysis output: {str(e)}")
            return {}

    def _calculate_differences(self,
                           user_analysis: Dict[str, Any],
                           ideal_analysis: Dict[str, Any]
                           ) -> List[Dict[str, Any]]:
        """2つの分析結果を比較し、差分一覧を返す"""
        differences = []

        metrics_to_compare = [
            ("bat_speed", "バットスピード"),
            ("weight_shift", "重心移動"),
            ("rotation_speed", "回転スピード"),
            ("impact_timing", "インパクトタイミング")
        ]

        for metric_key, metric_name in metrics_to_compare:
            user_value = user_analysis.get(metric_key, 0.0)
            ideal_value = ideal_analysis.get(metric_key, 0.0)
            gap = ideal_value - user_value

            differences.append({
                "metric": metric_name,
                "user_value": user_value,
                "ideal_value": ideal_value,
                "gap": gap,
                "percentage_diff": (gap / ideal_value * 100) if ideal_value != 0 else 0
            })

        return differences

    async def _generate_comparison_feedback(self,
                                        user_analysis: Dict[str, Any],
                                        ideal_analysis: Dict[str, Any],
                                        differences: List[Dict[str, Any]]) -> str:
        """comparison_prompt を使ってLLMに差分を分析してもらい、
        課題や改善提案をまとめたテキストを得る
        """
        comparison_input = {
            "user_analysis": user_analysis,
            "ideal_analysis": ideal_analysis,
            "differences": differences
        }
        input_text = json
    
    async def _generate_comparison_feedback(self,
                                        user_analysis: Dict[str, Any],
                                        ideal_analysis: Dict[str, Any],
                                        differences: List[Dict[str, Any]]) -> str:
        """comparison_prompt を使ってLLMに差分を分析してもらい、
        課題や改善提案をまとめたテキストを得る
        """
        comparison_input = {
            "user_analysis": user_analysis,
            "ideal_analysis": ideal_analysis,
            "differences": differences
        }
        input_text = json.dumps(comparison_input, ensure_ascii=False, indent=2)

        response = await self.llm.ainvoke(
            self.comparison_prompt.format(comparison_data=input_text)
        )
        return response.content