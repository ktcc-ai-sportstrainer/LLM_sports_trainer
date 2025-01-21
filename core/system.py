# core/system.py

import asyncio
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from agents.interactive_agent.agent import InteractiveAgent
from agents.modeling_agent.agent import ModelingAgent
from agents.goal_setting_agent.agent import GoalSettingAgent
from agents.plan_agent.agent import PlanAgent
from agents.search_agent.agent import SearchAgent
from agents.summarize_agent.agent import SummarizeAgent

from models.input.persona import Persona
from models.input.policy import TeachingPolicy
from models.internal.conversation import ConversationHistory

class SwingCoachingSystem:
    """
    システム全体を制御するクラス。
    エージェントを順番に呼び出し、結果をまとめる。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = ChatOpenAI(
            openai_api_key=config.get("openai_api_key", ""),
            model=config.get("model_name", "gpt-3.5-turbo"),
            temperature=0.7
        )

        # 各エージェントの初期化
        self.interactive_agent = InteractiveAgent(self.llm)
        self.modeling_agent = ModelingAgent(self.llm)
        self.goal_agent = GoalSettingAgent(self.llm)
        self.plan_agent = PlanAgent(self.llm)
        self.search_agent = SearchAgent(self.llm)
        self.summarize_agent = SummarizeAgent(self.llm)

    def run(
        self,
        persona_data: Dict[str, Any],
        policy_data: Dict[str, Any],
        video_path: str,
        existing_conversation: List[Any] = None
    ) -> Dict[str, Any]:
        # Persona, Policy, ConversationHistory の生成
        persona = Persona(**persona_data)
        policy = TeachingPolicy(**policy_data)
        conv_history = ConversationHistory(messages=existing_conversation or [])

        # 1. InteractiveAgent
        interactive_res = asyncio.run(
            self.interactive_agent.run(persona, policy, conv_history)
        )
        questions = interactive_res["content"]["questions"]

        # 2. ModelingAgent
        modeling_res = asyncio.run(
            self.modeling_agent.run(video_path)
        )
        # 解析結果
        motion_analysis_desc = modeling_res["content"]["analysis_description"]
        swing_analysis = modeling_res["content"]["swing_analysis"]
        issues_found = swing_analysis["issues_found"]  # 課題点

        # 3. GoalSettingAgent
        #   ここではダミーで "会話の洞察" をいくつか渡す
        conversation_insights = ["ユーザー回答から得られた追加の悩み", "気になるフォームのポイント"]
        goal_res = asyncio.run(
            self.goal_agent.run(
                persona=persona,
                policy=policy,
                conversation_insights=conversation_insights,
                motion_analysis=motion_analysis_desc
            )
        )
        final_goal = goal_res["content"]  # dict形式

        # 4. PlanAgent
        plan_res = asyncio.run(
            self.plan_agent.run(
                goal=final_goal,
                issues=issues_found
            )
        )

        # 5. SearchAgent (必要に応じてキーワード検索)
        sub_goals = final_goal.get("sub_goals", [])
        search_res = asyncio.run(
            self.search_agent.run(keywords=sub_goals)
        )

        # 6. SummarizeAgent
        summary_res = asyncio.run(
            self.summarize_agent.run(
                analysis=swing_analysis,
                goal=final_goal,
                plan=plan_res["content"]
            )
        )

        # 結果まとめ
        result = {
            "interactive_questions": questions,
            "motion_analysis": motion_analysis_desc,
            "goal_setting": final_goal,
            "training_plan": plan_res["content"],
            "search_results": search_res["content"],
            "final_summary": summary_res["content"]
        }
        return result
