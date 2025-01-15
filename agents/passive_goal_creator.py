from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from models.goal import Goal

class PassiveGoalCreator:
    def __init__(
            self,
            llm: ChatOpenAI
    ):
        self.llm = llm
    
    def run(self, query: str) -> Goal:
        prompt = ChatPromptTemplate.from_template(
            "ユーザーの入力を分析し、明確で実行可能な目標を生成してください。\n"
            "要件\n"
            "1. 目標は具体的かつ明確であり、実行可能なレベルで詳細化されている可能性があります。\n"
            "2. あなたが実行可能な行動は以下の通りです。\n"
            "  -インターネットを利用して、目標を達成するための調査を行う。\n"
            "  -ユーザーのためのレポートを生成する。\n"
            "3. 決して2.以外の行動を取ってはいけません。\n"
            "ユーザーの入力:{query}"
        )
        chain = prompt | self.llm.with_structured_output(Goal)
        return chain.invoke({"query": query})