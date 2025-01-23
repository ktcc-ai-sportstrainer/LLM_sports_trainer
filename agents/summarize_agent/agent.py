# agents/summarize_agent/agent.py

from typing import Dict, Any, List
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base import BaseAgent


class SummarizeAgent(BaseAgent):
    """
    システム全体の出力を最終的なコーチングレポートにまとめる
    """

    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompts = json.load(f)
        self.summary_prompt = ChatPromptTemplate.from_template(self.prompts["summary_prompt"])
        self.action_plan_prompt = ChatPromptTemplate.from_template(self.prompts["action_plan_prompt"])
        self.feedback_prompt = ChatPromptTemplate.from_template(self.prompts["feedback_prompt"])

    async def run(self, analysis: Dict[str, Any], goal: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        analysis: ModelingAgentの出力
        goal: GoalSettingAgentの出力
        plan: PlanAgentの出力
        """
        try:
            summary = await self._generate_summary(analysis, goal, plan)
            action_plan = await self._generate_action_plan(goal, plan)
            feedback = await self._generate_feedback(analysis, goal)

            final_report = {
                "summary": summary,
                "action_plan": action_plan,
                "feedback_points": feedback
            }
            return final_report

        except Exception as e:
            self.logger.log_error_details(error=e, agent=self.agent_name)
            return {}

    async def _generate_summary(self, analysis: Dict[str, Any], goal: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """
        全体サマリーを生成
        """
        response = await self.llm.ainvoke(
            self.summary_prompt.format_messages(
                analysis=json.dumps(analysis, ensure_ascii=False),
                goal=json.dumps(goal, ensure_ascii=False),
                plan=json.dumps(plan, ensure_ascii=False)
            )
        )
        try:
            summary_dict = json.loads(response.content)
            # 単純に keys, values を繋げて表示例
            lines = []
            for category, items in summary_dict.items():
                lines.append(f"▼{category}")
                for i in items:
                    lines.append(f"- {i}")
            return "\n".join(lines)
        except json.JSONDecodeError:
            return response.content

    async def _generate_action_plan(self, goal: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        アクションプランの生成（LLMを利用）
        """
        response = await self.llm.ainvoke(
            self.action_plan_prompt.format_messages(
                goal=json.dumps(goal, ensure_ascii=False),
                plan=json.dumps(plan, ensure_ascii=False)
            )
        )
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {"immediate_actions": [], "weekly_schedule": {}, "milestones": [], "raw_text": response.content}

    async def _generate_feedback(self, analysis: Dict[str, Any], goal: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        追加フィードバック
        """
        response = await self.llm.ainvoke(
            self.feedback_prompt.format_messages(
                analysis=json.dumps(analysis, ensure_ascii=False),
                goal=json.dumps(goal, ensure_ascii=False)
            )
        )
        try:
            data = json.loads(response.content)
            # critical/important/nice_to_have でまとめる例
            feedback_list = []
            for category in ["critical", "important", "nice_to_have"]:
                if category in data:
                    for item in data[category]:
                        feedback_list.append({
                            "priority": category,
                            "content": item["content"],
                            "reason": item.get("reason", ""),
                            "suggestion": item.get("suggestion", "")
                        })
            return feedback_list
        except json.JSONDecodeError:
            return [{"error": "could not parse feedback", "raw": response.content}]
