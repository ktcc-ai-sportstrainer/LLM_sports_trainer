from typing import Any, Dict, List
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base import BaseAgent
from models.input.persona import Persona
from models.input.policy import TeachingPolicy
from models.internal.goal import Goal

class SearchAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        self.search_tools = load_tools(["google-search", "web-browser"])
        
    async def run(
        self,
        search_requests: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        search_results = {}
        for request in search_requests:
            results = await self._execute_search(request)
            search_results[request["goal_id"]] = results
            
        analyzed_results = await self._analyze_search_results(search_results)
        return analyzed_results

    async def _execute_search(self, request: Dict[str, Any]) -> Dict[str, Any]:
        query = request["query"]
        category = request["category"]
        expected_info = request["expected_info"]
        
        raw_results = await self.search_tools["google-search"].arun(query)
        filtered_results = await self._filter_results(
            raw_results, category, expected_info
        )
        return filtered_results

    async def _filter_results(
        self,
        results: str,
        category: str,
        expected_info: str
    ) -> Dict[str, Any]:
        prompt = self.prompts["filter_prompt"].format(
            results=results,
            category=category,
            expected_info=expected_info
        )
        response = await self.llm.ainvoke(prompt)
        return json.loads(response.content)