from typing import Any, Dict, List
import json
import os
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import ChatOpenAI
from agents.base import BaseAgent

class SearchAgent(BaseAgent):
    """
    各種キーワードでGoogle検索を行い、その結果をPlanAgent等に提供するエージェント。
    """

    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        # prepare search tools
        self.search_tools = load_tools(["google-search"])
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def run(self, search_requests: str = None) -> str: # 戻り値を文字列に変更
        """
        search_requestsがNoneの場合は空の結果を返す
        """
        if not search_requests:
            return ""
            
        results_list = []
        for request in search_requests.split('\n'):
            single_result = await self._execute_search(request)
            print(f"フィルタの結果\n{single_result}")
            results_list.append(single_result)
        print(results_list)

        analyzed = await self._analyze_search_results(results_list)
        return analyzed # 文字列を返す

    async def _execute_search(self, request: str) -> str:
        """
        requestには {query, category, expected_info} 等が入る想定
        """
        query = request
        if not query:
            return ""

        # search_toolsをリストとして扱い、最初の要素を使用
        search_tool = self.search_tools[0]
        raw_results = await search_tool.ainvoke(input={"query": query}) # queryに変更
        print(f"{query}の検索結果{raw_results}")
        filtered = await self._filter_results(
            query, raw_results
        )
        return filtered

    async def _filter_results(self, query:str, results: str) -> str: # 戻り値を文字列に変更
        """
        LLMを使い、検索結果をカテゴリや期待情報でフィルタする。
        """
        prompt = self.prompts["filter_prompt"].format(
            query=query,
            results=results,
        )
        response = await self.llm.ainvoke(prompt)
        return response.content # 文字列を返す

    async def _analyze_search_results(self, results_list: List[str]) -> str: # 戻り値を文字列に変更
        """
        検索結果全体を俯瞰して要約するなどの処理
        """
        # result_analysis_promptを使用
        prompt = self.prompts["result_analysis_prompt"].format(results=results_list)
        resp = await self.llm.ainvoke(prompt)
        return resp.content # 文字列を返す