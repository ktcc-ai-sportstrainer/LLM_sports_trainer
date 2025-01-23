from datetime import datetime
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
    """
    システム全体を制御するクラス。
    persona_data, policy_data, user_video_path, ideal_video_path などを入力し、
    エージェント群(Interactive,Modeling,...)を順次呼び出して最終結果をまとめる。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = SystemLogger()

        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # LLMの初期化
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model=config.get("model_name", "gpt-4o-mini"),
            temperature=0.7
        )

        # エージェント辞書（後で初期化/更新）
        self.agents: Dict[str, Any] = {}

    async def run(
        self,
        persona_data: Dict[str, Any],
        policy_data: Dict[str, Any],
        user_video_path: str,
        ideal_video_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        1) InteractiveAgent で追加情報収集
        2) ModelingAgent で3D推定 & 分析
        3) GoalSettingAgent で目標設定
        4) PlanAgent で練習計画
        5) SearchAgent で補足情報検索
        6) SummarizeAgent で最終レポート
        """
        try:
            # Personaから身長を取得(例: "height" key)
            user_height = persona_data.get("height", 170.0)  # 未設定なら170cm仮定

            # 1. 各エージェントを初期化
            #   InteractiveAgent, ModelingAgent(user_height=...), ...
            interactive_agent = InteractiveAgent(self.llm)  # "mock"/"cli"/"streamlit"は後から変更するなど
            modeling_agent = ModelingAgent(self.llm, user_height=user_height)
            goal_agent = GoalSettingAgent(self.llm)
            plan_agent = PlanAgent(self.llm, SearchAgent(self.llm))
            summarize_agent = SummarizeAgent(self.llm)

            # 2. 実行フロー
            # 2-1) Interactive
            interactive_result = await interactive_agent.run(
                persona=persona_data,  # type: Personaのdict
                policy=policy_data     # type: Policyのdict
            )

            # 2-2) Modeling
            modeling_result = await modeling_agent.run(
                user_video_path, ideal_video_path
            )

            # 2-3) Goal Setting
            conversation_insights = interactive_result.get("interactive_insights", [])
            goal_result = await goal_agent.run(
                persona=persona_data,
                policy=policy_data,
                conversation_insights=conversation_insights,
                motion_analysis=modeling_result
            )

            # 2-4) Plan
            #   PlanAgentに目標やインサイト等を渡して計画を生成
            #   ただし、PlanAgent.run()のシグネチャは実装次第
            #   例: plan_agent.run(goal=goal_result["goals"], interactive_insights=conversation_insights)
            plan_output = await plan_agent.run(
                goal=goal_result.get("goals", {}),
                interactive_insights=conversation_insights
            )

            # 2-5) Summarize
            final_summary = await summarize_agent.run(
                analysis=modeling_result,
                goal=goal_result,
                plan=plan_output.get("plan", {})
            )

            # 3. 結果まとめ
            return {
                "interactive_questions": interactive_result.get("conversation_history", []),
                "motion_analysis": modeling_result,
                "goal_setting": goal_result,
                "training_plan": plan_output.get("plan", {}),
                "search_results": plan_output.get("plan", {}).get("references", {}),
                "final_summary": {
                    "summary": final_summary.get("summary", ""),
                    "execution_metrics": {
                        "status": "completed",
                        "error_count": 0,
                        "last_agent": "summary",
                        "total_time": 0.0  # 適宜計測
                    }
                }
            }

        except Exception as e:
            self.logger.log_error(f"System error: {str(e)}")
            raise
