from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class Reportor: 
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def run(self, query: str, response_definition: str, results: list[str]) -> str:
        prompt = ChatPromptTemplate.from_template(
            "与えられた目標:\n{query}\n\n"
            "調査結果:\n{results}\n\n"
            "与えられた目標に対し、調査結果を用いて、以下の指示に基づいてレスポンスを生成してください。"

            "{response_definition}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "query": query,
                "results": "\n\n".join(
                    f"Info{i+1}:\n{result}" for i, result in enumerate(results)
                ),
                "response_definition": response_definition, # これを追加
            }
        )