# agents/modeling_agent/agent.py

from typing import Any, Dict, List
import json
import os
import asyncio
import subprocess
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base import BaseAgent
from agents.modeling_agent.metrics.swing import SwingMetrics
from MotionAGFormer.JsonAnalist import analyze_json  

class ModelingAgent(BaseAgent):
    """
    3D姿勢推定とスイング分析を行うエージェント。
    1. user_video_pathを渡される
    2. vis.pyを呼び出して 3d_result.json を取得
    3. さらに jsonanalitst.py(analyze_json) を使って重心やバット速度等も計算
    4. LLMを使った言語化を行い、最終的に { user_analysis, ideal_analysis, comparison, ... } を出力
    """

    def __init__(self, llm: ChatOpenAI, user_height: float = 170.0):
        """
        user_height: ペルソナ情報の "height" (cm) を取得してコンストラクタへ渡す例
        """
        super().__init__(llm)
        self.swing_metrics = SwingMetrics()
        self.prompts = self._load_prompts()
        self.user_height = user_height

    async def run(self, user_video_path: str, ideal_video_path: str = None) -> Dict[str, Any]:
        result = {}
        try:
            # 1) ユーザー動画
            user_pose_json = await self._estimate_3d_pose(user_video_path, "user_3d.json")
            user_analysis = await self._analyze_swing(user_pose_json, "user")
            result["user_analysis"] = user_analysis

            # 2) 理想動画あれば
            if ideal_video_path:
                ideal_pose_json = await self._estimate_3d_pose(ideal_video_path, "ideal_3d.json")
                ideal_analysis = await self._analyze_swing(ideal_pose_json, "ideal")
                result["ideal_analysis"] = ideal_analysis

                # 比較
                comparison = await self._compare_swings(user_analysis, ideal_analysis)
                result["comparison"] = comparison
            else:
                # 単体分析
                general_analysis = await self._analyze_single_swing(user_analysis)
                result["general_analysis"] = general_analysis

            return result

        except Exception as e:
            self.logger.log_error_details(error=e, agent=self.agent_name)
            return {}

    async def _estimate_3d_pose(self, video_path: str, out_json_name: str) -> Dict[str, Any]:
        """
        Subprocessで vis.py を呼び出し、mp4 -> 3d_result.json を生成してもらい、
        そのJSONをdictに読み込んで返す。
        """
        output_dir = "./run/output_temp"  # 適宜
        os.makedirs(output_dir, exist_ok=True)

        # out_json名をフルパス化
        out_json_path = os.path.join(output_dir, out_json_name)

        # subprocess で実行
        cmd = [
            "python", 
            "MotionAGFormer/run/vis.py",
            "--video", video_path,
            "--out_json", out_json_name
        ]
        self.logger.log_info(f"Running vis.py cmd: {cmd}")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=".",  # カレントディレクトリ
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            err_msg = stderr.decode()
            raise RuntimeError(f"vis.py failed: {err_msg}")

        # vis.py が標準出力に最終JSONをprintしているはず
        stdout_str = stdout.decode()
        try:
            data_dict = json.loads(stdout_str)
        except json.JSONDecodeError:
            data_dict = {}

        # もしファイルからも読みたいなら
        if os.path.exists(out_json_path):
            with open(out_json_path, "r", encoding="utf-8") as f:
                file_data = json.load(f)
                # file_data が stdout で受け取った data_dict と同じ想定
                data_dict = file_data

        return data_dict

    async def _analyze_swing(self, pose_json: Dict[str, Any], label: str) -> Dict[str, Any]:
        """
        pose_json は vis.py 標準出力 or ファイルから読み込んだ 3D座標構造 ( video_file, frames=[...] )。
        さらに jsonanalitst.py の analyze_json を呼んで、バット速度や重心などを取得。
        """
        # まずは3D座標をファイルに書き出し、一時jsonにする
        temp_path = f"./run/temp_{label}_3d_input.json"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump({
                "frames": [
                    {
                        "frame_index": fdata["frame_index"],
                        "coordinates": [
                            {
                                "joint_name": "", 
                                "x": 0.0, "y": 0.0, "z": 0.0
                            }
                            # ↓↓↓
                            # ここでは実際に joint_names と fdata["coordinates"] の整合を取り、 
                            # {joint_name, x, y, z} 構造に変換
                            for _ in range(17)
                        ]
                    } for fdata in pose_json.get("frames", [])
                ]
            }, f, indent=2)

        # ここで "temp_path" を JsonAnalistに渡す
        analysis_result = analyze_json(temp_path, verbose=False)
        # analysis_result = {
        #   "idealgravity": [...],
        #   "judge": [...],
        #   "speed": ...,
        #   ...
        # }

        # LLMへのプロンプト例:
        # if self.prompts["metrics_analysis_prompt"]: 
        #   prompt = ...
        #   response = await self.llm.ainvoke(prompt)
        #   ...

        # ここではシンプルに analysis_result をまとめて返す
        return {
            "pose_json": pose_json,    # 3D座標(ノーマライズ後)
            "analyst_result": analysis_result
        }

    async def _compare_swings(
        self,
        user_analysis: Dict[str, Any],
        ideal_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        2動画のanalysisをLLMに比較させる想定
        """
        prompt = self.prompts["comparison_prompt"].format(
            user_analysis=json.dumps(user_analysis, ensure_ascii=False),
            ideal_analysis=json.dumps(ideal_analysis, ensure_ascii=False)
        )
        response = await self.llm.ainvoke(prompt)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"comparison_text": response.content}

    async def _analyze_single_swing(self, user_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        1動画だけのとき: 一般論比較
        """
        prompt = self.prompts["single_swing_analysis_prompt"].format(
            swing_analysis=json.dumps(user_analysis, ensure_ascii=False)
        )
        response = await self.llm.ainvoke(prompt)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"general_analysis_text": response.content}

    def _load_prompts(self) -> Dict[str, str]:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return json.load(f)
