import json
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from agents.base import BaseAgent

class GoalSettingAgent(BaseAgent):
    """
    各種情報を元に、部員に合ったバッティング改善の目標を設定するエージェント。
    """

    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        # 同階層にある prompts.json を読み込む想定
        import os
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def run(
        self,
        persona: Dict[str, Any],
        policy: Dict[str, Any],
        conversation_insights: List[Any],
        motion_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        情報をまとめ、LLMに目標を作ってもらう。
        """
        try:
            # goals_promptを使用
            prompt_template = self.prompts["goals_prompt"]
            prompt = prompt_template.format(
                persona=json.dumps(persona, ensure_ascii=False),
                policy=json.dumps(policy, ensure_ascii=False),
                insights=json.dumps(conversation_insights, ensure_ascii=False),
                analysis=json.dumps(motion_analysis, ensure_ascii=False)
            )
            response = await self.llm.ainvoke(prompt)
            # LLMがJSON形式で返す想定
            try:
                goal_data = json.loads(response.content)
            except json.JSONDecodeError:
                goal_data = {"primary_goal": {}, "sub_goals": []}

            return {
                "goals": goal_data
            }

        except Exception as e:
            self.logger.log_error_details(error=e, agent=self.agent_name)
            return {}
