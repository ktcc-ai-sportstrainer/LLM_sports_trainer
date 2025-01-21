# agents/summarize_agent/agent.py

from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agents.base import BaseAgent

class SummarizeAgent(BaseAgent):
    """
    各エージェントの最終結果を受け取り、要点をまとめて出力するエージェント。
    """

    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        self.prompt = ChatPromptTemplate.from_template(
            """以下に、スイング解析・目標設定・練習計画などの情報が含まれています。
            これらを踏まえて、選手が理解しやすいように最終的なコーチング方針をまとめてください。

            【スイング解析結果】
            {analysis}

            【目標設定】
            {goal}

            【練習計画】
            {plan}

            該当の情報がない場合は省略して構いません。

            なるべく箇条書き等を使い、明確かつシンプルにまとめてください。
            """
        )

    async def run(
        self,
        analysis: Dict[str, Any],
        goal: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        分析結果, 目標, 練習計画をまとめてユーザーへ返すテキストを生成。
        """
        # analysis, goal, plan はそれぞれdict
        # 文字列整形
        analysis_str = str(analysis)
        goal_str = str(goal)
        plan_str = str(plan)

        response = await self.llm.ainvoke(
            self.prompt.format(
                analysis=analysis_str,
                goal=goal_str,
                plan=plan_str
            )
        )
        final_summary = response.content.strip()

        return self.create_output(
            output_type="final_summary",
            content={"summary": final_summary}
        ).dict()
