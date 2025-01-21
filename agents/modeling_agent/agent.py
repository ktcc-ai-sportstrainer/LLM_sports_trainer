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

        # 既存の analysis_prompt, visualization_prompt を読み込む
        self.analysis_prompt = ChatPromptTemplate.from_template(prompts["analysis_prompt"])
        self.visualization_prompt = ChatPromptTemplate.from_template(prompts["visualization_prompt"])
        
        # 新たに comparison_prompt を追加 (prompts.json に追記済み)
        self.comparison_prompt = ChatPromptTemplate.from_template(prompts["comparison_prompt"])

    async def run(self,
                  user_video_path: str,
                  ideal_video_path: str) -> Dict[str, Any]:
        """
        2つの動画（ユーザーのスイングと理想のスイング）を
        MotionAGFormer + JsonAnalist.py で解析し、
        差分比較のフィードバックを加えた結果を返す。
        """

        # 1. 各動画を 3D推定
        user_json_path = await self._estimate_3d_poses(user_video_path, "output/user_swing")
        ideal_json_path = await self._estimate_3d_poses(ideal_video_path, "output/ideal_swing")

        # 2. JsonAnalist.py による分析
        user_analysis = await self._analyze_3d_json(user_json_path)
        ideal_analysis = await self._analyze_3d_json(ideal_json_path)

        # 3. 差分算出
        differences = self._calculate_differences(user_analysis, ideal_analysis)

        # 4. LLMに比較用データを投げてフィードバック生成
        comparison_feedback = await self._generate_comparison_feedback(
            user_analysis, ideal_analysis, differences
        )

        # 5. 結果をまとめて返却
        return {
            "user_analysis": user_analysis,
            "ideal_analysis": ideal_analysis,
            "differences": differences,
            "comparison_feedback": comparison_feedback
        }

    async def _estimate_3d_poses(self, video_path: str, output_dir: str) -> str:
        """
        1つの動画に対する3D姿勢推定を行い、生成されたJSONファイルのパスを返す。
        - 内部的には MotionAGFormer/run/vis.py をサブプロセス呼び出ししている想定
        - 出力先を output_dir/3d_result.json とし、それを返す
        """
        os.makedirs(output_dir, exist_ok=True)

        # vis.py に引数を渡して実行する例
        cmd = [
            "python", "MotionAGFormer/run/vis.py",
            "--video", video_path,
            "--output", output_dir
        ]
        subprocess.run(cmd, check=True)

        # vis.py 側で "3d_result.json" という名前で保存される想定
        json_path = os.path.join(output_dir, "3d_result.json")
        return json_path

    async def _analyze_3d_json(self, json_file_path: str) -> Dict[str, Any]:
        """
        JsonAnalist.py による解析を行う。
        - "--input" に 3D推定jsonを指定し、標準出力に結果が JSON で出る想定
        - Pythonでサブプロセスを呼び出し、stdoutを json.loads して返す
        """
        cmd = ["python", "MotionAGFormer/JsonAnalist.py", "--input", json_file_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # JsonAnalist.py が stdout で JSON を返す想定
        analysis_result = json.loads(proc.stdout)
        return analysis_result

    def _calculate_differences(self,
                               user_analysis: Dict[str, Any],
                               ideal_analysis: Dict[str, Any]
                               ) -> List[Dict[str, Any]]:
        """
        2つの分析結果（user / ideal）を比較し、差分を返す。
        - metricごとに user, ideal の値を見て 'gap' を計算
        - 必要に応じて suggestions など付与しても良い
        """
        differences = []

        # 例: バット速度の比較
        user_speed = user_analysis.get("bat_speed", 0.0)
        ideal_speed = ideal_analysis.get("bat_speed", 0.0)
        gap_speed = ideal_speed - user_speed
        differences.append({
            "metric": "bat_speed",
            "user_value": user_speed,
            "ideal_value": ideal_speed,
            "gap": gap_speed
        })

        # 例: 重心移動
        user_weight_shift = user_analysis.get("weight_shift", 0.0)
        ideal_weight_shift = ideal_analysis.get("weight_shift", 0.0)
        gap_weight = ideal_weight_shift - user_weight_shift
        differences.append({
            "metric": "weight_shift",
            "user_value": user_weight_shift,
            "ideal_value": ideal_weight_shift,
            "gap": gap_weight
        })

        # 他に hips_rotation_speed, shoulder_rotation_speed など続けて比較
        # ...

        return differences

    async def _generate_comparison_feedback(self,
                                            user_analysis: Dict[str, Any],
                                            ideal_analysis: Dict[str, Any],
                                            differences: List[Dict[str, Any]]) -> str:
        """
        comparison_prompt を用いて2つのスイング比較に基づく
        フィードバック（課題/強み/改善提案）を LLM に出してもらう。
        """
        # 比較用データを整形
        comparison_input = {
            "user_analysis": user_analysis,
            "ideal_analysis": ideal_analysis,
            "differences": differences
        }
        input_text = json.dumps(comparison_input, ensure_ascii=False, indent=2)

        # comparison_prompt を呼び出し
        response = await self.llm.ainvoke(
            self.comparison_prompt.format(comparison_data=input_text)
        )
        return response.content