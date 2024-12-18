from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from models.task import Task


class Executor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = [TavilySearchResults(max_results=3)]
        self.base_agent = create_react_agent(self.llm, self.tools)

    def run(self, task: Task) -> str:
        result = self.base_agent.invoke(
            {
                "messages": [
                    (
                        "system",
                        (
                            f"あなたは{task.role.name}です。\n"
                            f"説明:{task.role.description}\n"
                            f"主要なスキル:{','.join(task.role.key_skills)}\n"
                            "あなたの役割に基づいて、与えられたタスクを最高の能力で遂行してくください。"
                        ),
                    ),
                    (
                        "human",
                        f"以下のタスクを実行してください:\n\n{task.description}",
                    ),
                ]
            }
        )
        return result["messages"][-1].content
