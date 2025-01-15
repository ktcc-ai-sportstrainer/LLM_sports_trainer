from langchain_openai import ChatOpenAI
from agents.query_decomposer import QueryDecomposer
from models.task import Task
from models.decomposed_tasks import DecomposedTasks


class Planner:
    def __init__(self, llm:ChatOpenAI):
        self.query_decomposer = QueryDecomposer(llm=llm)

    def run(self, query:str) -> list[Task]:
        decomposed_tasks: DecomposedTasks = self.query_decomposer.run(query=query)
        return [Task(description=task) for task in decomposed_tasks.values]