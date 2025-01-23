# agents/plan_agent/agent.py

import json
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from agents.base import BaseAgent
from agents.search_agent.agent import SearchAgent  # 同階層 or 相対 importに注意

class PlanAgent(BaseAgent):
    """
    目標を達成するためのタスクや練習プランを作成する。
    必要に応じてSearchAgentで情報を収集し、再度プランに反映する。
    """

    def __init__(self, llm: ChatOpenAI, search_agent: SearchAgent):
        super().__init__(llm)
        self.search_agent = search_agent
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        import os
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def run(
        self,
        goal: Dict[str, Any],
        interactive_insights: List[str] = None
    ) -> Dict[str, Any]:
        """
        1. 目標から不足情報を特定
        2. SearchAgentで情報収集
        3. 収集結果を使ってプランを生成
        """
        try:
            # 1. 検索クエリ生成
            search_queries = await self._generate_search_queries(goal)
            # 2. SearchAgentで検索実行
            search_results = await self.search_agent.run(search_queries)
            # 3. 検索結果 + goal から最終プラン生成
            tasks, schedule = await self._create_plan(goal, search_results)
            return {
                "plan": {
                    "tasks": tasks,
                    "schedule": schedule,
                    "references": search_results
                }
            }
        except Exception as e:
            self.logger.log_error_details(error=e, agent=self.agent_name)
            return {}

    async def _generate_search_queries(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        LLMに検索クエリを作らせる。
        """
        prompt_template = self.prompts["search_queries_prompt"]
        prompt = prompt_template.format(
            goals=json.dumps(goal, ensure_ascii=False)
        )
        response = await self.llm.ainvoke(prompt)
        try:
            data = json.loads(response.content)
            return data.get("queries", [])
        except json.JSONDecodeError:
            return []

    async def _create_plan(self, goal: Dict[str, Any], search_results: Dict[str, Any]) -> (List[Dict[str, Any]], Dict[str, Any]):
        """
        検索結果と目標からトレーニングプランをLLMで生成
        """
        prompt = self.prompts["task_generation_prompt"].format(
            goals=json.dumps(goal, ensure_ascii=False),
            search_results=json.dumps(search_results, ensure_ascii=False)
        )
        response = await self.llm.ainvoke(prompt)
        try:
            tasks_data = json.loads(response.content).get("tasks", [])
        except json.JSONDecodeError:
            tasks_data = []

        # 次にスケジュール生成
        schedule_prompt = self.prompts["schedule_prompt"].format(
            tasks=json.dumps(tasks_data, ensure_ascii=False),
            goals=json.dumps(goal, ensure_ascii=False)
        )
        schedule_response = await self.llm.ainvoke(schedule_prompt)
        try:
            schedule_data = json.loads(schedule_response.content)
        except json.JSONDecodeError:
            schedule_data = {}

        return tasks_data, schedule_data
