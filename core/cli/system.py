from typing import Dict, Any, Optional
import os

from langchain_openai import ChatOpenAI
from agents import (
    InteractiveAgent,
    ModelingAgent,
    GoalSettingAgent,
    PlanAgent,
    SearchAgent,
    SummarizeAgent
)
from core.base.logger import SystemLogger
from core.base.state import create_initial_state, SystemState

class SwingCoachingSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = SystemLogger()

        # LLMの初期化
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model=config.get("model_name", "gpt-4"),
            temperature=0.7
        )

        # エージェントの初期化
        self.agents = {
            "interactive": InteractiveAgent(self.llm, mode="cli"),
            "modeling": ModelingAgent(self.llm),
            "goal_setting": GoalSettingAgent(self.llm),
            "search": SearchAgent(self.llm),
            "plan": None,  # SearchAgentに依存するので後で初期化
            "summarize": SummarizeAgent(self.llm)
        }
        
        # PlanAgentの初期化
        self.agents["plan"] = PlanAgent(self.llm, self.agents["search"])

    async def run(
        self,
        persona_data: Dict[str, Any],
        policy_data: Dict[str, Any],
        user_video_path: Optional[str] = None,
        ideal_video_path: Optional[str] = None,
        user_pose_json: Optional[str] = None,
        ideal_pose_json: Optional[str] = None
    ) -> Dict[str, Any]:
        # 初期状態の作成
        initial_state = create_initial_state(
            persona_data, 
            policy_data, 
            user_video_path, 
            ideal_video_path,
            user_pose_json,
            ideal_pose_json
        )
        state = SystemState(initial_state)
        
        try:
            # 1. ユーザーとの対話
            self.logger.log_info("Starting interactive session...", agent="interactive")
            interactive_result = await self.agents["interactive"].run(
                persona=persona_data,
                policy=policy_data
            )
            state.update({"conversation": interactive_result})

            # 2. 動作分析
            self.logger.log_info("Starting modeling...", agent="modeling")
            modeling_result = await self.agents["modeling"].run(
                user_video_path=user_video_path,
                ideal_video_path=ideal_video_path,
                user_pose_json=user_pose_json,
                ideal_pose_json=ideal_pose_json
            )
            motion_analysis_result = modeling_result.get("analysis_result", "")
            state.update({"motion_analysis": motion_analysis_result})

            # 3. 目標設定
            self.logger.log_info("Setting goals...", agent="goal_setting")
            conversation_insights = interactive_result.get("interactive_insights", [])
            goal_result = await self.agents["goal_setting"].run(
                persona=persona_data,
                policy=policy_data,
                conversation_insights=conversation_insights,
                motion_analysis=motion_analysis_result
            )
            state.update({"goals": goal_result.get("goal_setting_result", "")})

            # 4. トレーニングプラン作成
            self.logger.log_info("Creating training plan...", agent="plan")
            plan_result = await self.agents["plan"].run(
                goal=goal_result.get("goal_setting_result", ""),
                motion_analysis=motion_analysis_result
            )
            state.update({"plan": plan_result})

            # 5. 関連情報の検索
            self.logger.log_info("Searching relevant information...", agent="search")
            search_result = await self.agents["search"].run(plan_result)
            state.update({"search_results": search_result})

            # 6. 最終サマリーの生成
            self.logger.log_info("Generating final summary...", agent="summarize")
            final_summary = await self.agents["summarize"].run(
                analysis=motion_analysis_result,
                goal=goal_result.get("goal_setting_result", ""),
                plan=plan_result
            )
            state.update({"summary": final_summary})

            # 7. 結果の整形と返却
            return {
                "interactive_result": interactive_result,
                "motion_analysis": motion_analysis_result,
                "goal_setting": goal_result.get("goal_setting_result", ""),
                "search_results": search_result,
                "training_plan": plan_result,
                "final_summary": final_summary
            }

        except Exception as e:
            self.logger.log_error_details(error=e, agent="system")
            raise

    async def _estimate_3d_pose(self, video_path: str) -> Dict[str, Any]:
        """3D姿勢推定を実行"""
        try:
            cmd = [
                "python",
                "MotionAGFormer/run/vis.py",
                "--video", video_path
            ]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            stdout_str = stdout.decode()
            stderr_str = stderr.decode()

            self.logger.log_debug(f"Pose estimation stdout: {stdout_str}")
            if stderr_str:
                self.logger.log_warning(f"Pose estimation stderr: {stderr_str}")

            # JSONデータの抽出
            try:
                json_start = stdout_str.find('{')
                json_end = stdout_str.rfind('}') + 1
                if json_start >= 0 and json_end > 0:
                    json_str = stdout_str[json_start:json_end]
                    return json.loads(json_str)
                return {}
            except json.JSONDecodeError as e:
                self.logger.log_error(f"JSON decode error: {e}")
                return {}

        except Exception as e:
            self.logger.log_error_details(error=e, agent="system")
            raise