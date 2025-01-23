from typing import Any, Dict, List
import json
import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base import BaseAgent
from agents.modeling_agent.metrics.swing import SwingMetrics

class ModelingAgent(BaseAgent):
   def __init__(self, llm: ChatOpenAI):
       super().__init__(llm)
       self.swing_metrics = SwingMetrics()
       self.prompts = self._load_prompts()

   async def run(self, user_video_path: str, ideal_video_path: str = None) -> Dict[str, Any]:
       try:
           # ユーザーのスイング分析
           user_pose_data = await self._estimate_3d_pose(user_video_path)
           user_analysis = await self._analyze_swing(user_pose_data)
           
           # 理想のスイング動画が与えられた場合は比較分析
           if ideal_video_path:
               ideal_pose_data = await self._estimate_3d_pose(ideal_video_path)
               ideal_analysis = await self._analyze_swing(ideal_pose_data)
               comparison = await self._compare_swings(user_analysis, ideal_analysis)
               return {
                   "user_analysis": user_analysis,
                   "ideal_analysis": ideal_analysis, 
                   "comparison": comparison
               }
           
           # 単一動画の場合は一般的な基準との比較
           else:
               general_analysis = await self._analyze_single_swing(user_analysis)
               return {
                   "user_analysis": user_analysis,
                   "general_analysis": general_analysis
               }

       except Exception as e:
           self.logger.error(f"Error in ModelingAgent: {str(e)}")
           return {}

   async def _estimate_3d_pose(self, video_path: str) -> Dict[str, Any]:
       """MotionAGFormerを使用して3D姿勢推定を実行"""
       process = await asyncio.create_subprocess_exec(
           "python",
           "MotionAGFormer/run/vis.py",
           "--video", video_path,
           stdout=asyncio.subprocess.PIPE,
           stderr=asyncio.subprocess.PIPE
       )
       stdout, stderr = await process.communicate()
       
       if process.returncode != 0:
           raise Exception(f"3D pose estimation failed: {stderr.decode()}")
           
       return json.loads(stdout.decode())

   async def _analyze_swing(self, pose_data: Dict[str, Any]) -> Dict[str, Any]:
       """スイングの各指標を計算"""
       phases = self.swing_metrics.detect_swing_phases(pose_data)
       bat_speed = self.swing_metrics.calculate_bat_speed(pose_data, phases["contact"])
       rotation_speed = self.swing_metrics.calculate_rotation_speed(pose_data, "hips")
       rotation_sequence = self.swing_metrics.evaluate_rotation_sequence(pose_data, phases)
       weight_shift = self.swing_metrics.analyze_weight_shift(pose_data, phases)
       swing_plane = self.swing_metrics.calculate_swing_plane(pose_data)
       
       return {
           "phases": phases,
           "metrics": {
               "bat_speed": bat_speed,
               "rotation_speed": rotation_speed,
               "rotation_sequence": rotation_sequence,
               "weight_shift": weight_shift,
               "swing_plane": swing_plane
           }
       }

   async def _compare_swings(
       self,
       user_analysis: Dict[str, Any],
       ideal_analysis: Dict[str, Any]
   ) -> Dict[str, Any]:
       """ユーザーと理想のスイングを比較"""
       prompt = self.prompts["comparison_prompt"].format(
           user_analysis=json.dumps(user_analysis, ensure_ascii=False),
           ideal_analysis=json.dumps(ideal_analysis, ensure_ascii=False)
       )
       
       response = await self.llm.ainvoke(prompt)
       return json.loads(response.content)

   async def _analyze_single_swing(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
       """一般的な基準と比較して分析"""
       prompt = self.prompts["single_swing_analysis_prompt"].format(
           swing_analysis=json.dumps(analysis, ensure_ascii=False)
       )
       response = await self.llm.ainvoke(prompt)
       return json.loads(response.content)

   def _load_prompts(self) -> Dict[str, str]:
       prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
       with open(prompt_path, "r", encoding="utf-8") as f:
           return json.load(f)