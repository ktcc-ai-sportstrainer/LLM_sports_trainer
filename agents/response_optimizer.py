from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ResponseOptimizer:
    def __init__(
            self,
            llm: ChatOpenAI
    ):
        self.llm = llm
    
    def run(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """あなたはAIエージェントシステムのレスポンス最適化スペシャリストです。与えられた目標に対して、エージェントが
                    目標にあったレスポンスを返すためのレスポンス仕様を策定してください。"""
                ),
                (
                    "human",
                    "以下の手順に従って、レスポンス最適化プロンプトを作成してください:\n\n"
                    "1. 目標分析:\n"
                    "提示された目標を分析し、主要な要素や意図を特定してください。\n\n"
                    "2. レスポンス仕様の策定:\n"
                    """目標達成のための適切なレスポンス仕様を考案してください。トーン、構造、内容の焦点などを考慮に
                    入れてください。\n\n"""
                    "3. 具体的な指示の作成:\n"
                    """事前に収集された情報から、ユーザーの期待に沿ったレスポンスをするために必要な、AIエージェントに対する
                    明確で実行可能な指示を作成してください。あなたの指示によってAIエージェントが実行可能なのは、既に調査済み
                    の結果をまとめることだけです。インターネットへのアクセスはできません。\n\n"""
                    "4. 例の提供:\n"
                    "可能であれば、目標に沿ったレスポンスの例を1つ以上含めてください。\n\n"
                    "5. 評価基準の設定:\n"
                    "レスポンスの効果を測定するための基準を定義してください。\n\n"
                    "以下の構造でレスポンス最適化プロンプトを出力してください。:\n\n"
                    "目標分析:\n"
                    "[ここに目標の分析結果を記入]\n\n"
                    "レスポンス仕様:\n"
                    "[ここに策定されたレスポンスの仕様を記入]\n\n"
                    "AIエージェントへの指示:\n"
                    "[ここにAIエージェントへの具体的な指示を記入]\n\n"
                    "レスポンス例:\n"
                    "[ここにレスポンス例を記入]\n\n"
                    "評価基準:\n"
                    "[ここに評価基準を記入]\n\n"
                    "では、以下の目標に対するレスポンス最適化プロンプトを作成してください:\n"
                    "{query}",
                ),
           ]
       )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
