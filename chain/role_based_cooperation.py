# 必要なimport文
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agents.executor import Executor

# agents以下をimport
from agents.passive_goal_creator import PassiveGoalCreator
from agents.planner import Planner
from agents.prompt_optimizer import PromptOptimizer
from agents.reportor import Reportor
from agents.response_optimizer import ResponseOptimizer
from agents.role_assigner import RoleAssigner
from core.agent_state import AgentState  # AgentStateをimport
from models.goal import Goal
from models.optimized_goal import OptimizedGoal


class RoleBasedCooperation:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.passive_goal_creator = PassiveGoalCreator(llm=llm)
        self.prompt_optimizer = PromptOptimizer(llm=llm)
        self.response_optimizer = ResponseOptimizer(llm=llm)
        self.planner = Planner(llm=llm)
        self.role_assigner = RoleAssigner(llm=llm)
        self.executor = Executor(llm=llm)
        self.reporter = Reportor(llm=llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("goal_setting", self._goal_setting)
        workflow.add_node("planner", self._plan_tasks)
        workflow.add_node("role_assigner", self._assign_roles)
        workflow.add_node("executor", self._execute_task)
        workflow.add_node("reporter", self._generate_report)

        workflow.set_entry_point("goal_setting")

        workflow.add_edge("goal_setting", "planner")
        workflow.add_edge("planner", "role_assigner")
        workflow.add_edge("role_assigner", "executor")
        workflow.add_conditional_edges(
            "executor",
            lambda state: state.current_task_index < len(state.tasks),
            {True: "executor", False: "reporter"},
        )

        workflow.add_edge("reporter", END)

        return workflow.compile()

    def _goal_setting(self, state: AgentState) -> dict[str, Any]:
        goal: Goal = self.passive_goal_creator.run(query=state.query)
        optimized_goal: OptimizedGoal = self.prompt_optimizer.run(query=goal.text)
        optimized_response: str = self.response_optimizer.run(query=optimized_goal.text)

        return {
            "optimized_goal": optimized_goal.text,
            "optimized_response": optimized_response,
        }

    def _plan_tasks(self, state: AgentState) -> dict[str, Any]:
        tasks = self.planner.run(query=state.optimized_goal)
        return {"tasks": tasks}

    def _assign_roles(self, state: AgentState) -> dict[str, Any]:
        tasks_with_roles = self.role_assigner.run(tasks=state.tasks)
        return {"tasks": tasks_with_roles}

    def _execute_task(self, state: AgentState) -> dict[str, Any]:
        current_task = state.tasks[state.current_task_index]
        result = self.executor.run(task=current_task)
        return {
            "results": [result],
            "current_task_index": state.current_task_index + 1,
        }

    def _generate_report(self, state: AgentState) -> dict[str, Any]:
        report = self.reporter.run(
            query=state.optimized_goal,
            response_definition=state.optimized_response,
            results=state.results,
        )
        print(report)
        return {"final_report": report}

    # def run(self, query: str) -> str:
    #     initial_state = AgentState(query=query)

    #     final_state = self.graph.invoke(initial_state)
    #     return final_state["final_report"]
