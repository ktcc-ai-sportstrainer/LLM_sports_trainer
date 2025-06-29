from typing import Dict, Any, Optional, Tuple
import os
import asyncio
import json
from langchain_google_genai import ChatGoogleGenerativeAI

from core.base.logger import SystemLogger
from core.webui.state import WebUIState
from core.webui.media import VideoDisplay
from agents import (
    InteractiveAgent,
    ModelingAgent,
    GoalSettingAgent,
    PlanAgent,
    SearchAgent,
    SummarizeAgent
)

class WebUISwingCoachingSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = SystemLogger()
        self.video_display = VideoDisplay()
        self.interactive_enabled = True

        # LLMの初期化
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        self.llm = ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model=config.get("model_name", "gemini-1.5-pro-latest"),
            temperature=0.7
        )

        # エージェントの初期化
        self.setup_agents()

    def setup_agents(self):
        """エージェントの初期化"""
        self.agents = {
            "interactive": InteractiveAgent(self.llm, mode="streamlit"),
            "modeling": ModelingAgent(self.llm),
            "goal_setting": GoalSettingAgent(self.llm),
            "search": SearchAgent(self.llm),
            "plan": None,  # SearchAgentに依存するので後で初期化
            "summarize": SummarizeAgent(self.llm)
        }
        
        # PlanAgentの初期化
        self.agents["plan"] = PlanAgent(self.llm, self.agents["search"])

    async def process_video(self, video_path: str) -> Tuple[str, str, str]:
        """
        動画処理を実行し、3D姿勢推定結果とビジュアライゼーション動画を返す
        Returns:
            Tuple[str, str, str]: (pose_json_path, visualization_video_path, visualization_json_path)
        """
        try:
            # 動画名から出力ディレクトリを設定
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = f'./run/output/{video_name}/'
            os.makedirs(output_dir, exist_ok=True)
            
            # 出力JSONのパス
            pose_json_path = os.path.join(output_dir, "3d_result.json")

            # MotionAGFormerの実行
            cmd = [
                "python",
                "MotionAGFormer/run/vis.py",
                "--video", video_path,
                "--gpu", "0"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if stderr:
                self.logger.log_warning(f"Pose estimation stderr: {stderr.decode()}")

            if process.returncode != 0:
                raise RuntimeError(f"Pose estimation failed: {stderr.decode()}")

            # vis.pyの出力規則に従ってパスを設定
            vis_video_path = os.path.join(output_dir, f"{video_name}.mp4")
            vis_json_path = os.path.join(output_dir, "visualization_data.json")

            # 結果のJSONをvisualization_data.jsonにコピー
            if os.path.exists(pose_json_path):
                with open(pose_json_path, 'r') as f:
                    vis_data = json.load(f)
                with open(vis_json_path, 'w') as f:
                    json.dump(vis_data, f, indent=4)

            if not os.path.exists(vis_video_path):
                raise FileNotFoundError(f"Visualization video not generated: {vis_video_path}")

            # 表示用の動画パスを生成
            display_video_path = self.video_display.prepare_video_display(vis_video_path)

            return pose_json_path, display_video_path, vis_json_path

        except Exception as e:
            self.logger.log_error_details(error=e, agent="system")
            raise

    async def run(
        self,
        persona_data: Dict[str, Any],
        policy_data: Dict[str, Any],
        user_pose_json: Optional[str] = None,
        ideal_pose_json: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """システムの実行（WebUI用）"""
        try:
            results = {}
            state = kwargs.get('state')

            # モデリング実行
            self.logger.log_info("Starting modeling...", agent="modeling")
            modeling_result = await self.agents["modeling"].run(
                user_pose_json=user_pose_json,
                ideal_pose_json=ideal_pose_json
            )
            motion_analysis_result = modeling_result.get("analysis_result", "")
            if state:
                state.update({"motion_analysis": motion_analysis_result})
            results["modeling"] = modeling_result

            # インタラクティブモードが有効な場合のみ実行
            if self.interactive_enabled:
                self.logger.log_info("Starting interactive session...", agent="interactive")
                interactive_result = await self.agents["interactive"].run(
                    persona=persona_data,
                    policy=policy_data
                )
                if state:
                    state.update({"conversation": interactive_result})
                results["interactive"] = interactive_result
                conversation_insights = interactive_result.get("interactive_insights", [])
            else:
                conversation_insights = []

            # 目標設定
            self.logger.log_info("Setting goals...", agent="goal_setting")
            goal_result = await self.agents["goal_setting"].run(
                persona=persona_data,
                policy=policy_data,
                conversation_insights=conversation_insights,
                motion_analysis=motion_analysis_result
            )
            if state:
                state.update({"goals": goal_result.get("goal_setting_result", "")})
            results["goal_setting"] = goal_result

            # トレーニングプラン作成
            self.logger.log_info("Creating training plan...", agent="plan")
            plan_result = await self.agents["plan"].run(
                goal=goal_result.get("goal_setting_result", ""),
                motion_analysis=motion_analysis_result
            )
            if state:
                state.update({"plan": plan_result})
            results["training_plan"] = plan_result

            # 関連情報の検索
            self.logger.log_info("Searching relevant information...", agent="search")
            search_result = await self.agents["search"].run(plan_result)
            if state:
                state.update({"search_results": search_result})
            results["search_results"] = search_result

            # 最終サマリーの生成
            self.logger.log_info("Generating final summary...", agent="summarize")
            final_summary = await self.agents["summarize"].run(
                analysis=motion_analysis_result,
                goal=goal_result.get("goal_setting_result", ""),
                plan=plan_result
            )
            if state:
                state.update({"summary": final_summary})
            results["final_summary"] = final_summary

            return results

        except Exception as e:
            self.logger.log_error_details(error=e, agent="system")
            raise

    def cleanup(self):
        """リソースのクリーンアップ"""
        self.video_display.cleanup()