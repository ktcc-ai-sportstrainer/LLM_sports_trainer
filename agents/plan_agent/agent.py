from typing import Any, Dict, List
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base import BaseAgent
from agents.search_agent import SearchAgent
from models.internal.goal import Goal
from models.internal.plan import TrainingPlan, TrainingTask

class PlanAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI, search_agent: SearchAgent):
        super().__init__(llm)
        self.search_agent = search_agent

    async def run(
        self,
        goals: Dict[str, Any],
        search_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        # 1. 検索の実行
        search_results = await self.search_agent.run(search_queries)
        
        # 2. 検索結果を基にタスク生成
        tasks = await self._generate_tasks(goals, search_results)
        
        # 3. タスクのスケジューリング
        schedule = await self._create_schedule(tasks, goals)
        
        return {
            "tasks": tasks,
            "schedule": schedule,
            "references": search_results
        }

    async def _generate_tasks(
        self,
        goals: Dict[str, Any],
        search_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        prompt = self.prompts["task_generation_prompt"].format(
            goals=json.dumps(goals, ensure_ascii=False),
            search_results=json.dumps(search_results, ensure_ascii=False)
        )
        response = await self.llm.ainvoke(prompt)
        return json.loads(response.content)

    async def _create_schedule(
        self,
        tasks: List[Dict[str, Any]],
        goals: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt = self.prompts["schedule_prompt"].format(
            tasks=json.dumps(tasks, ensure_ascii=False),
            goals=json.dumps(goals, ensure_ascii=False)
        )
        response = await self.llm.ainvoke(prompt)
        return json.loads(response.content)