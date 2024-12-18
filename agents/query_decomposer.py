import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from models.decomposed_tasks import DecomposedTasks


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_data = datetime.datetime.now().strftime("%Y-%m-%d")

    def run(self, query: str) -> DecomposedTasks:
        prompt = ChatPromptTemplate.from_template(
            f"CURRENT_DATE: {self.current_data}\n"
            "-----\n"
            """タスク: 与えられた目標を具体的で実行可能なタスクに分解してください。特に重要なタスクは練習メニューを作る
            ことと、ユーザーの悩みに応じた人間味のある応援を考えることです。\n"""
            "要件:\n"
            "1. 以下の行動だけで目標を達成すること。決して指定された行動をとらないこと。\n"
            "   - インターネットを利用して、目標を達成するための調査を行う。\n"
            "2. 各タスクは具体的かつ詳細に記載されており、単独で実行並びに検証可能な情報を含めること。一切抽象的な表現を含まないこと。\n"
            "3. タスクは実行可能な順序でリスト化すること。\n"
            "4. 特に重要なタスクは最優先で必ず実行してください。\n"
            "5. タスクは日本語で出力すること。\n"
            "6. タスクは3個から5個で出力してください。"
            "目標: {query}"
        )
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.invoke({"query": query})  # これを追加
