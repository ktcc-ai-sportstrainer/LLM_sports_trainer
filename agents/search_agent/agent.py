# agents/search_agent/agent.py

import time
from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from agents.base import BaseAgent

class SearchAgent(BaseAgent):
    """
    外部のリソースやAPIを検索し、提案に使える追加情報を取得するエージェント。
    ここではダミー実装している。
    """

    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)

    async def run(self, keywords: List[str]) -> Dict[str, Any]:
        """
        指定されたキーワードを使って情報検索を行い、結果を返すダミー。
        """
        # 本来はここで外部API呼び出しなど
        time.sleep(1)

        # ダミーの返答: "keywords"の内容を単純にレスポンスに入れる
        dummy_resources = [
            {
                "title": f"{kw} Drills",
                "url": f"https://example.com/drills?search={kw}",
                "description": f"A set of practice drills focused on {kw}"
            }
            for kw in keywords
        ]

        return self.create_output(
            output_type="search_results",
            content={"resources": dummy_resources}
        ).dict()
