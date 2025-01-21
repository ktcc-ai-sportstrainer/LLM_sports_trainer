import os
import json
import subprocess
from typing import Dict, Any, List
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

        # 新しく comparison_prompt を追加
        self.comparison_prompt = ChatPromptTemplate.from_template(prompts["comparison_prompt"])

    async def run(self,
                  user_video_path: str,
                  ideal_video_path: str) -> Dict[str, Any]:
        """
        2つの動画（ユーザーのスイングと理想のスイング）を
        MotionAGFormer + JsonAnalist で解析し、差分比較。
        """
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

    async def _estimate_3d_poses(self, video_path: str, output_dir: str) -> str:
        """
        MotionAGFormer/run/vis.py を呼び出して 3D姿勢推定json を生成
        """
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            "python",
            "MotionAGFormer/run/vis.py",
            "--video", video_path,
            "--output", output_dir
        ]
        subprocess.run(cmd, check=True)

        # vis.py が "3d_result.json" として出力
        json_path = os.path.join(output_dir, "3d_result.json")
        return json_path

    async def _analyze_3d_json(self, json_file_path: str) -> Dict[str, Any]:
        """
        JsonAnalist.py で解析し、stdoutに出る結果を JSON ロードして返す
        """
        cmd = ["python", "MotionAGFormer/JsonAnalist.py", "--input", json_file_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        analysis_result = json.loads(proc.stdout)
        return analysis_result

    def _calculate_differences(self,
                               user_analysis: Dict[str, Any],
                               ideal_analysis: Dict[str, Any]
                               ) -> List[Dict[str, Any]]:
        """
        2つの分析結果を比較し、差分一覧を返す
        """
        differences = []

        # 例: bat_speed
        user_speed = user_analysis.get("bat_speed", 0.0)
        ideal_speed = ideal_analysis.get("bat_speed", 0.0)
        gap_speed = ideal_speed - user_speed
        differences.append({
            "metric": "bat_speed",
            "user_value": user_speed,
            "ideal_value": ideal_speed,
            "gap": gap_speed
        })

        # 例: weight_shift
        user_ws = user_analysis.get("weight_shift", 0.0)
        ideal_ws = ideal_analysis.get("weight_shift", 0.0)
        gap_ws = ideal_ws - user_ws
        differences.append({
            "metric": "weight_shift",
            "user_value": user_ws,
            "ideal_value": ideal_ws,
            "gap": gap_ws
        })

        # ほか hips_rotation_speed など好きに追加
        return differences

    async def _generate_comparison_feedback(self,
                                            user_analysis: Dict[str, Any],
                                            ideal_analysis: Dict[str, Any],
                                            differences: List[Dict[str, Any]]) -> str:
        """
        comparison_prompt を使ってLLMに差分を分析してもらい、
        課題や改善提案をまとめたテキスト(またはJSON)を得る
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
