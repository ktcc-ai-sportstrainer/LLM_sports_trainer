# core/system.py
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
from core.logger import SystemLogger
from core.state import StateManager, create_initial_state
from utils.json_handler import JSONHandler

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
            model=config.get("model_name", "gpt-4o-mini"),
            temperature=0.7
        )

        # エージェントの初期化（順序を考慮）
        self.agents = {
            "interactive": InteractiveAgent(self.llm),
            "modeling": ModelingAgent(self.llm),
            "goal_setting": GoalSettingAgent(self.llm),
            "search": SearchAgent(self.llm),  # SearchAgentを先に初期化
        }
        
        # PlanAgentはSearchAgentに依存するので後で初期化
        self.agents["plan"] = PlanAgent(self.llm, self.agents["search"])
        
        # SummarizeAgentの初期化
        self.agents["summarize"] = SummarizeAgent(self.llm)

    async def run(
        self,
        persona_data: Dict[str, Any],
        policy_data: Dict[str, Any],
        user_video_path: Optional[str] = None,
        ideal_video_path: Optional[str] = None,
        user_pose_json: Optional[str] = None,
        ideal_pose_json: Optional[str] = None
    ) -> Dict[str, Any]:
        # StateManagerの初期化をtryの外に移動
        initial_state = create_initial_state(persona_data, policy_data, user_video_path, ideal_video_path)
        self.state_manager = StateManager(initial_state)
        
        try:
            # 1. ユーザーとの対話
            self.logger.log_info("Starting interactive session...", agent="interactive")
            interactive_result = await self.agents["interactive"].run(
                persona=persona_data,
                policy=policy_data
            )
            self.state_manager.update_state("interactive", {"conversation": interactive_result})

            # 2. 動作分析
            self.logger.log_info("Starting modeling...", agent="modeling")
            modeling_result = await self.agents["modeling"].run(
                user_video_path=user_video_path,
                ideal_video_path=ideal_video_path,
                user_pose_json=user_pose_json,
                ideal_pose_json=ideal_pose_json
            )
            # 3D姿勢推定結果は保持するが、後続のエージェントには文字列を渡す
            motion_analysis_result = modeling_result.get("analysis_result", "")

            # ModelingAgentの結果を'motion_analysis'キーに格納
            self.state_manager.update_state("modeling", {
                "motion_analysis": motion_analysis_result
            })
            

            # 3. 目標設定
            self.logger.log_info("Setting goals...", agent="goal_setting")
            conversation_insights = interactive_result.get("interactive_insights", [])

            goal_result = await self.agents["goal_setting"].run(
                persona=persona_data,
                policy=policy_data,
                conversation_insights=conversation_insights,
                motion_analysis=motion_analysis_result
            )
            self.state_manager.update_state("goal_setting", {"goals": goal_result.get("goal_setting_result", "")})

            # 4. トレーニングプランの作成
            self.logger.log_info("Creating training plan...", agent="plan")
            plan_result = await self.agents["plan"].run(
                goal=goal_result.get("goal_setting_result", ""),
                motion_analysis=motion_analysis_result,
            )
            self.state_manager.update_state("plan", {"plan": plan_result})

            # 5. 関連情報の検索
            self.logger.log_info("Searching relevant information...", agent="search")
            search_queries = plan_result # ここでplan_resultを使用
            search_result = await self.agents["search"].run(
                search_requests=search_queries
            )
            self.state_manager.update_state("search", {"search_results": search_result})

            # 6. 最終サマリーの生成
            self.logger.log_info("Generating final summary...", agent="summarize")
            final_summary = await self.agents["summarize"].run(
                analysis=motion_analysis_result,
                goal=goal_result.get("goal_setting_result", ""),
                plan=plan_result
            )
            self.state_manager.update_state("summarize", {"summary": final_summary})

            # 最終サマリーをテキストファイルに保存
            self.logger.log_info("Saving final summary to file...", agent="system")
            summary_file_path = os.path.join(self.config.get("output_dir", "output"), "final_summary.txt")
            os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
            with open(summary_file_path, "w", encoding="utf-8") as f:
                f.write(final_summary)

            # 最終的な結果の整形
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