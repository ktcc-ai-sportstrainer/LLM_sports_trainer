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
                    user_pose_data = json.load(f)
                    user_analysis_text = await self._analyze_swing(user_pose_data, "user")
            elif user_video_path:
                # 動画から3D姿勢推定（従来の処理）
                user_pose_data = await self._estimate_3d_pose(user_video_path, "user_3d.json")
                user_analysis_text = await self._analyze_swing(user_pose_data, "user")
            else:
                raise ValueError("Either user_video_path or user_pose_json must be provided")

            # 理想スイングの分析（ある場合）
            ideal_analysis_text = "" # 初期値を空文字列に変更
            if ideal_pose_json:
                with open(ideal_pose_json, 'r') as f:
                    ideal_pose_data = json.load(f)
                    ideal_analysis_text = await self._analyze_swing(ideal_pose_data, "ideal")
            elif ideal_video_path:
                ideal_pose_data = await self._estimate_3d_pose(ideal_video_path, "ideal_3d.json")
                ideal_analysis_text = await self._analyze_swing(ideal_pose_data, "ideal")

            if ideal_analysis_text:
                # 比較分析
                comparison_text = await self._compare_swings(user_analysis_text, ideal_analysis_text)
                return {"analysis_result": comparison_text} # 文字列を返す
            else:
                # 一般論に基づく分析
                general_analysis_text = await self._analyze_single_swing(user_analysis_text)
                return {"analysis_result": general_analysis_text} # 文字列を返す

        except Exception as e:
            self.logger.log_error_details(error=e, agent=self.agent_name)
            return {"analysis_result": f"エラーが発生しました: {e}"} # エラーメッセージを返す

        
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

    async def _analyze_swing(self, pose_json: Dict[str, Any], label: str) -> str: # 戻り値を文字列に変更
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

            # 解析結果を文字列化して返す
            return json.dumps(analysis_result, indent=2, ensure_ascii=False)

        except Exception as e:
            self.logger.log_error_details(error=e, agent=self.agent_name)
            return f"エラーが発生しました: {e}"

    def _get_metrics_description(self) -> str:
        """
        analyze_json関数で計算される指標の説明を返す
        """
        description = """
        - idealgravity: 各フレームにおける全身の重心座標のリスト
        - judge: 各フレームでバットがストライクゾーン内にあるかどうかの判定結果（真偽値）のリスト
        - speed: インパクト時の推定バットスピード（最大値）
        - speed_list: 各フレームにおける推定バットスピードのリスト
        - speed_list_len: speed_listの長さ（フレーム数）
        - max_speed_index: バットスピードが最大となるフレームのインデックス
        """
        return description

    async def _compare_swings(self, user_analysis: str, ideal_analysis: str) -> str:
        """
        2動画のanalysisをLLMに比較させる
        """
        metrics_description = self._get_metrics_description()
        prompt = self.prompts["comparison_prompt"].format(
            user_analysis=user_analysis,
            ideal_analysis=ideal_analysis,
            metrics_description=metrics_description # 指標の説明を追加
        )
        response = await self.llm.ainvoke(prompt)
        return response.content


    async def _analyze_single_swing(self, user_analysis: str) -> str:
        """
        1動画だけのとき: 一般論比較
        """
        metrics_description = self._get_metrics_description()
        prompt = self.prompts["single_swing_analysis_prompt"].format(
            swing_analysis=user_analysis,
            metrics_description=metrics_description # 指標の説明を追加
        )
        response = await self.llm.ainvoke(prompt)
        return response.content

    def _load_prompts(self) -> Dict[str, str]:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return json.load(f)