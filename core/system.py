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
        try:
            # 1. ユーザーとの対話
            self.logger.log_info("Starting interactive session...")
            interactive_result = await self.agents["interactive"].run(
                persona=persona_data,
                policy=policy_data
            )

            # 2. 動作分析
            modeling_result = await self.agents["modeling"].run(
                user_video_path=user_video_path,
                ideal_video_path=ideal_video_path,
                user_pose_json=user_pose_json,
                ideal_pose_json=ideal_pose_json
            )

            # 3. 目標設定
            self.logger.log_info("Setting goals...")
            conversation_insights = interactive_result.get("interactive_insights", [])
            goal_result = await self.agents["goal_setting"].run(
                persona=persona_data,
                policy=policy_data,
                conversation_insights=conversation_insights,
                motion_analysis=modeling_result
            )

            # 4. 関連情報の検索
            self.logger.log_info("Searching relevant information...")
            search_result = await self.agents["search"].run(
                search_queries=goal_result.get("search_queries", [])
            )

            # 5. トレーニング計画の作成
            self.logger.log_info("Creating training plan...")
            plan_result = await self.agents["plan"].run(
                goal=goal_result.get("goals", {}),
                search_results=search_result
            )

            # 6. 最終サマリーの生成
            self.logger.log_info("Generating final summary...")
            final_summary = await self.agents["summarize"].run(
                analysis=modeling_result,
                goal=goal_result,
                plan=plan_result
            )

            # 結果の整形と返却
            return {
                "interactive_questions": interactive_result.get("conversation_history", []),
                "motion_analysis": modeling_result,
                "goal_setting": goal_result,
                "search_results": search_result,
                "training_plan": plan_result,
                "final_summary": final_summary
            }

        except Exception as e:
            self.logger.log_error(f"System error: {str(e)}")
            raise