from typing import Dict, Any, List
import json
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base import BaseAgent


class SummarizeAgent(BaseAgent):
    """
    システム全体の出力を最終的なコーチングレポートにまとめる
    """

    def __init__(self, llm: ChatGoogleGenerativeAI):
        super().__init__(llm)
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompts = json.load(f)
        self.summary_prompt = self.prompts["summary_prompt"]
        self.action_plan_prompt = self.prompts["action_plan_prompt"]
        self.feedback_prompt = self.prompts["feedback_prompt"]

    async def run(self, analysis: str, goal: str, plan: str, persona: Dict[str, Any], policy: Dict[str, Any]) -> str:
        """
        analysis: ModelingAgentの出力
        goal: GoalSettingAgentの出力
        plan: PlanAgentの出力
        persona: 選手情報
        policy: 指導方針
        """
        try:
            summary = await self._generate_summary(analysis, goal, plan)
            action_plan = await self._generate_action_plan(goal, plan, json.dumps(persona), json.dumps(policy))
            feedback = await self._generate_feedback(analysis, goal)

            final_report = f"## コーチングレポート\n\n{summary}\n\n## アクションプラン\n\n{action_plan}\n\n## フィードバック\n\n{feedback}"
            return final_report

        except Exception as e:
            self.logger.log_error_details(error=e, agent=self.agent_name)
            return ""

    async def _generate_summary(self, analysis: str, goal: str, plan: str) -> str:
        """
        全体サマリーを生成
        """
        response = await self.llm.ainvoke(
            self.summary_prompt.format(
                analysis=analysis,
                goal=goal,
                plan=plan
            )
        )
        return response.content

    async def _generate_action_plan(self, goal: str, plan: str, persona: str, policy: str) -> str:
        """
        アクションプランの生成（LLMを利用）
        Parameters:
            goal: 目標
            plan: 練習計画
            persona: 選手情報（JSON文字列）
            policy: 指導方針（JSON文字列）
        """
        response = await self.llm.ainvoke(
            self.action_plan_prompt.format(
                goal=goal,
                plan=plan,
                persona=persona,
                policy=policy
            )
        )
        return response.content

    async def _generate_feedback(self, analysis: str, goal: str) -> str:
        """
        追加フィードバック
        """
        response = await self.llm.ainvoke(
            self.feedback_prompt.format(
                analysis=analysis,
                goal=goal
            )
        )
        return response.content