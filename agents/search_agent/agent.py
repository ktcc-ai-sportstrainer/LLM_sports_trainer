from typing import Any, Dict, List
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain_community.utilities import GoogleSearchAPIWrapper

from agents.base import BaseAgent
from models.internal.goal import Goal
from models.internal.plan import TrainingTask

class SearchAgent(BaseAgent):
    """練習タスクに関連する具体的な練習メニューや指導のポイントを検索して提案するエージェント"""

    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        
        # プロンプトの読み込み
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
            
        self.search_prompt = ChatPromptTemplate.from_template(prompts["search_prompt"])
        self.synthesis_prompt = ChatPromptTemplate.from_template(prompts["synthesis_prompt"])
        self.drill_prompt = ChatPromptTemplate.from_template(prompts["drill_prompt"])

        # 検索ツールの設定
        self.search = GoogleSearchAPIWrapper()
        self.search_tool = Tool(
            name="Google Search",
            description="Search Google for recent information about baseball training drills",
            func=self.search.run
        )

    async def run(
        self,
        tasks: List[Dict[str, Any]],
        player_level: str
    ) -> Dict[str, Any]:
        """タスクに関連する練習メニューの検索と生成"""
        detailed_tasks = []
        
        for task in tasks:
            # 1. 関連情報の検索
            search_results = await self._search_relevant_info(task, player_level)
            
            # 2. 検索結果の統合
            synthesized_info = await self._synthesize_info(search_results, task)
            
            # 3. 具体的な練習メニューの生成
            training_menu = await self._generate_training_menu(
                task, synthesized_info, player_level
            )
            
            detailed_task = {
                **task,
                "training_menu": training_menu
            }
            detailed_tasks.append(detailed_task)
        
        return self.create_output(
            output_type="detailed_training_tasks",
            content={"tasks": detailed_tasks}
        ).dict()

    async def _search_relevant_info(
        self,
        task: Dict[str, Any],
        player_level: str
    ) -> List[str]:
        """タスクに関連する情報を検索"""
        # 検索クエリの生成
        response = await self.llm.ainvoke(
            self.search_prompt.format(
                task=json.dumps(task, ensure_ascii=False),
                player_level=player_level
            )
        )
        
        search_queries = json.loads(response.content)["queries"]
        
        # 各クエリで検索を実行
        all_results = []
        for query in search_queries:
            try:
                results = await self.search_tool.arun(query)
                all_results.append(results)
            except Exception as e:
                print(f"Search error for query '{query}': {str(e)}")
                continue
        
        return all_results

    async def _synthesize_info(
        self,
        search_results: List[str],
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """検索結果を統合して構造化"""
        response = await self.llm.ainvoke(
            self.synthesis_prompt.format(
                search_results="\n".join(search_results),
                task=json.dumps(task, ensure_ascii=False)
            )
        )
        
        return json.loads(response.content)

    async def _generate_training_menu(
        self,
        task: Dict[str, Any],
        synthesized_info: Dict[str, Any],
        player_level: str
    ) -> Dict[str, Any]:
        """具体的な練習メニューを生成"""
        response = await self.llm.ainvoke(
            self.drill_prompt.format(
                task=json.dumps(task, ensure_ascii=False),
                synthesized_info=json.dumps(synthesized_info, ensure_ascii=False),
                player_level=player_level
            )
        )
        
        return json.loads(response.content)

    def _validate_search_results(self, results: List[str]) -> bool:
        """検索結果の妥当性をチェック"""
        if not results:
            return False
            
        # 最小限の結果数
        if len(results) < 3:
            return False
            
        # 各結果の最小文字数
        for result in results:
            if len(result.strip()) < 50:
                return False
                
        return True

    def _validate_training_menu(self, menu: Dict[str, Any]) -> bool:
        """生成された練習メニューの妥当性をチェック"""
        required_keys = {"drills", "progression_steps", "total_time"}
        
        # 必須キーの存在チェック
        if not all(key in menu for key in required_keys):
            return False
            
        # ドリルの存在チェック
        if not menu["drills"] or not isinstance(menu["drills"], list):
            return False
            
        # 各ドリルの妥当性チェック
        for drill in menu["drills"]:
            if not all(key in drill for key in {"name", "description", "duration"}):
                return False
                
        return True