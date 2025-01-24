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
        try:
            # StateManagerの初期化
            initial_state = create_initial_state(persona_data, policy_data, user_video_path, ideal_video_path)
            self.state_manager = StateManager(initial_state)
            
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
            
            if "user_analysis" in modeling_result:
                user_analysis_result = modeling_result["user_analysis"].get("analyst_result", {})
            else:
                user_analysis_result = {}

            # ModelingAgentの結果を'motion_analysis'キーに格納
            self.state_manager.update_state("modeling", {
                "motion_analysis": {
                    "user_analysis": modeling_result.get("user_analysis",{}),
                    "ideal_analysis": modeling_result.get("ideal_analysis",{}),
                    "comparison": modeling_result.get("comparison", ""),
                    "general_analysis": modeling_result.get("general_analysis", "")
                }
            })
            

            # 3. 目標設定
            self.logger.log_info("Setting goals...", agent="goal_setting")
            conversation_insights = interactive_result.get("interactive_insights", [])

            # ModelingAgentの結果から必要な情報を抽出
            if "comparison" in modeling_result:
                motion_analysis_result = modeling_result["comparison"]
            elif "general_analysis" in modeling_result:
                motion_analysis_result = modeling_result["general_analysis"]
            else:
                motion_analysis_result = ""

            goal_result = await self.agents["goal_setting"].run(
                persona=persona_data,
                policy=policy_data,
                conversation_insights=conversation_insights,
                motion_analysis=motion_analysis_result
            )
            self.state_manager.update_state("goal_setting", {"goals": goal_result.get("goal_setting_result", "")})

            # 4. 関連情報の検索
            self.logger.log_info("Searching relevant information...", agent="search")
            # 検索クエリ自体はGoalSettingAgentの出力から直接取得せず、文字列として扱う
            search_queries_str = goal_result.get("goal_setting_result", "")
            search_result = await self.agents["search"].run(
                search_queries=search_queries_str
            )
            self.state_manager.update_state("search", {"search_results": search_result})

            # 5. トレーニング計画の作成
            self.logger.log_info("Creating training plan...", agent="plan")
            plan_result = await self.agents["plan"].run(
                goal=goal_result.get("goal_setting_result", ""),  # goal_setting_resultキーから取得
                motion_analysis=motion_analysis_result,
                search_results=search_result
            )
            self.state_manager.update_state("plan", {"plan": plan_result})

            # 6. 最終サマリーの生成
            self.logger.log_info("Generating final summary...", agent="summarize")
            final_summary = await self.agents["summarize"].run(
                analysis=motion_analysis_result,
                goal=goal_result.get("goal_setting_result", ""),  # goal_setting_resultキーから取得
                plan=plan_result
            )
            self.state_manager.update_state("summarize", {"summary": final_summary})

            # 最終的な結果の整形
            return {
                "interactive_result": interactive_result,
                "motion_analysis": modeling_result,
                "goal_setting": goal_result.get("goal_setting_result", ""),
                "search_results": search_result,
                "training_plan": plan_result,
                "final_summary": final_summary
            }

        except Exception as e:
            self.logger.log_error_details(error=e, agent="system")
            raise