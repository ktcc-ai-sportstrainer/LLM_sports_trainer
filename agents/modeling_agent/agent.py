from typing import Any, Dict, List, Optional
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
    def __init__(self, llm: ChatOpenAI, user_height: float = 170.0):
        super().__init__(llm)
        self.swing_metrics = SwingMetrics()
        self.prompts = self._load_prompts()
        self.user_height = user_height

    async def run(
        self,
        user_video_path: Optional[str] = None,
        ideal_video_path: Optional[str] = None,
        user_pose_json: Optional[str] = None,
        ideal_pose_json: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            # ユーザーのスイング分析
            if user_pose_json:
                # JSONから直接読み込み
                with open(user_pose_json, 'r') as f:
                    user_analysis = await self._analyze_swing(json.load(f), "user")
            elif user_video_path:
                # 動画から3D姿勢推定（従来の処理）
                user_pose_data = await self._estimate_3d_pose(user_video_path, "user_3d.json")
                user_analysis = await self._analyze_swing(user_pose_data, "user")
            else:
                raise ValueError("Either user_video_path or user_pose_json must be provided")

            # 理想スイングの分析（ある場合）
            ideal_analysis = None
            if ideal_pose_json:
                with open(ideal_pose_json, 'r') as f:
                    ideal_analysis = await self._analyze_swing(json.load(f), "ideal")
            elif ideal_video_path:
                ideal_pose_data = await self._estimate_3d_pose(ideal_video_path, "ideal_3d.json")
                ideal_analysis = await self._analyze_swing(ideal_pose_data, "ideal")

            result = {
                "user_analysis": user_analysis
            }

            if ideal_analysis:
                result["ideal_analysis"] = ideal_analysis
                comparison = await self._compare_swings(user_analysis, ideal_analysis)
                result["comparison"] = comparison
            else:
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
        """pose_jsonからjoint_namesとの整合性を取る処理を追加"""
        temp_path = f"./run/temp_{label}_3d_input.json"
        
        # joint_names との対応を定義
        joint_mapping = {
            "Hip": "Hip", "RHip": "RHip", "RKnee": "RKnee", "RAnkle": "RAnkle",
            "LHip": "LHip", "LKnee": "LKnee", "LAnkle": "LAnkle",
            "Spine": "Spine", "Thorax": "Thorax", "Neck/Nose": "Neck/Nose", "Head": "Head",
            "LShoulder": "LShoulder", "LElbow": "LElbow", "LWrist": "LWrist",
            "RShoulder": "RShoulder", "RElbow": "RElbow", "RWrist": "RWrist"
        }

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({
                    "frames": [
                        {
                            "frame_index": fdata["frame_index"],
                            "coordinates": [
                                {
                                    "joint_name": joint_name,
                                    "x": fdata["coordinates"][i][0] if i < len(fdata["coordinates"]) else 0.0,
                                    "y": fdata["coordinates"][i][1] if i < len(fdata["coordinates"]) else 0.0,
                                    "z": fdata["coordinates"][i][2] if i < len(fdata["coordinates"]) else 0.0
                                }
                                for i, joint_name in enumerate(joint_mapping.keys())
                            ]
                        }
                        for fdata in pose_json.get("frames", [])
                    ]
                }, f, indent=2)

            # JsonAnalistでの分析実行
            analysis_result = analyze_json(temp_path, user_height=self.user_height, verbose=False)
            
            return {
                "pose_json": pose_json,
                "analyst_result": analysis_result
            }

        except Exception as e:
            self.logger.log_error_details(error=e, agent=self.agent_name)
            return {
                "pose_json": pose_json,
                "analyst_result": {},
                "error": str(e)
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
