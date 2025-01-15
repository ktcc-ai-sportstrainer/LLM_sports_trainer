from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from models.optimized_goal import OptimizedGoal

class PromptOptimizer:
    def __init__(
            self,
            llm: ChatOpenAI
    ):
        self.llm = llm
    
    def run(self, query: str) -> OptimizedGoal:
        prompt = ChatPromptTemplate.from_template(
            """あなたは目標設定の専門家です。以下のユーザーの悩みとプロフィールからSMART原則(Specific: 具体的、Measurable: 測定可能、
            Achievable: 達成目標、Relavant: 関連性が高い、 Time-bound: 期限がある、)に基づいて最適化してください。\n\n"""
            "ユーザーの悩みとプロフィール:\n"
            "{query}\n\n"
            "指示:\n"
            "1. 元の目標を分析し、不足している要素や改善点を特定してください。\n"
            "2. あなたが実行可能なのは以下の行動だけです。\n"
            "  -インターネットを利用して、目標を達成するための調査を行う。\n"
            "  -ユーザーのためのレポートを生成する。\n"
            "3. SMART原則の各要素を考慮しながら、目標を具体的かつ詳細に記載してください。\n"
            "  -一切抽象的な表現を含んではいけません。\n"
            "  -必ずすべての単語が実行か可能かつ具体的であることを確認してください。\n"
            "4. 目標の達成度を測定する方法を具体的かつ詳細に記載してください。\n"
            "5. 元の目標で期限が指定されていない場合は、期限を考慮する必要はありません。\n"
            "6. REMEMBER: 決して2.以外の行動を取ってはいけません。"
        )
        chain = prompt | self.llm.with_structured_output(OptimizedGoal)
        return chain.invoke({"query": query})