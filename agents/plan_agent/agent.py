import json
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from agents.base import BaseAgent
from agents.search_agent.agent import SearchAgent

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
        goal: str, # 文字列に変更
        motion_analysis: str # 文字列を追加
    ) -> str: # 戻り値を文字列に変更
        """
        1. 目標から不足情報を特定
        2. SearchAgentで情報収集
        3. 収集結果を使ってプランを生成
        """
        try:
            # 1. 検索クエリ生成
            search_queries = await self._generate_search_queries(goal, motion_analysis)
            # 2. SearchAgentで検索実行
            search_results = await self.search_agent.run(search_queries)
            # 3. 検索結果 + goal から最終プラン生成
            plan = await self._create_plan(goal, motion_analysis, search_results)
            return plan # 文字列を返す

        except Exception as e:
            self.logger.log_error_details(error=e, agent=self.agent_name)
            return "" # エラー時は空文字列を返す

    async def _generate_search_queries(self, goal: str, motion_analysis: str) -> str: # 戻り値を文字列に変更
        """
        LLMに検索クエリを作らせる。
        """
        prompt_template = self.prompts["search_queries_prompt"]
        prompt = prompt_template.format(
            goals=goal,
            analysis=motion_analysis # 分析結果追加
        )
        response = await self.llm.ainvoke(prompt)
        return response.content # 文字列を返す

    async def _create_plan(self, goal: str, motion_analysis: str, search_results: str) -> str: # 戻り値を文字列に変更
        """
        検索結果と目標からトレーニングプランをLLMで生成
        """
        prompt = self.prompts["task_generation_prompt"].format(
            goals=goal,
            analysis=motion_analysis, # 分析結果追加
            search_results=search_results
        )
        response = await self.llm.ainvoke(prompt)
        return response.content # 文字列を返す