from typing import Any, Dict, List
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base import BaseAgent
from models.internal.goal import Goal
from models.internal.plan import TrainingPlan, TrainingTask

class PlanAgent(BaseAgent):
    """目標に基づいて具体的な練習計画を生成するエージェント"""

    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        
        # プロンプトの読み込み
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
            
        self.plan_prompt = ChatPromptTemplate.from_template(prompts["plan_prompt"])
        self.task_prompt = ChatPromptTemplate.from_template(prompts["task_prompt"])
        self.progression_prompt = ChatPromptTemplate.from_template(prompts["progression_prompt"])

    async def run(
        self,
        goal: Goal,
        issues: List[str]
    ) -> Dict[str, Any]:
        """練習計画の生成"""
        # 1. 全体計画の生成
        overall_plan = await self._generate_overall_plan(goal, issues)
        
        # 2. 具体的なタスクの生成
        tasks = await self._generate_tasks(overall_plan, goal)
        
        # 3. 進行計画の生成
        progression_path = await self._generate_progression_path(tasks, goal)
        
        # 4. TrainingPlan オブジェクトの生成
        training_plan = TrainingPlan(
            tasks=tasks,
            progression_path=progression_path,
            required_time=self._calculate_total_time(tasks)
        )
        
        return self.create_output(
            output_type="training_plan",
            content=training_plan.dict()
        ).dict()

    async def _generate_overall_plan(
        self,
        goal: Goal,
        issues: List[str]
    ) -> Dict[str, Any]:
        """全体的な練習計画の生成"""
        response = await self.llm.ainvoke(
            self.plan_prompt.format(
                primary_goal=goal.primary_goal,
                sub_goals=goal.sub_goals,
                metrics=goal.metrics,
                issues=issues
            )
        )
        
        plan_dict = json.loads(response.content)
        return plan_dict

    async def _generate_tasks(
        self,
        overall_plan: Dict[str, Any],
        goal: Goal
    ) -> List[TrainingTask]:
        """具体的な練習タスクの生成"""
        tasks = []
        for area in overall_plan["training_areas"]:
            response = await self.llm.ainvoke(
                self.task_prompt.format(
                    training_area=area,
                    goal=goal.dict()
                )
            )
            
            task_dict = json.loads(response.content)
            task = TrainingTask(
                title=task_dict["title"],
                description=task_dict["description"],
                duration=task_dict["duration"],
                focus_points=task_dict["focus_points"],
                equipment=task_dict["equipment"]
            )
            tasks.append(task)
        
        return tasks

    async def _generate_progression_path(
        self,
        tasks: List[TrainingTask],
        goal: Goal
    ) -> str:
        """段階的な上達計画の生成"""
        response = await self.llm.ainvoke(
            self.progression_prompt.format(
                tasks=json.dumps([t.dict() for t in tasks], ensure_ascii=False),
                goal=goal.dict()
            )
        )
        
        path_dict = json.loads(response.content)
        return path_dict["progression_path"]

    def _calculate_total_time(self, tasks: List[TrainingTask]) -> str:
        """全タスクの所要時間を計算"""
        total_minutes = 0
        for task in tasks:
            # 時間文字列（例：「30分」）から数値を抽出
            time_str = task.duration
            try:
                minutes = int(''.join(filter(str.isdigit, time_str)))
                total_minutes += minutes
            except ValueError:
                continue
        
        # 時間と分に変換
        hours = total_minutes // 60
        minutes = total_minutes % 60
        
        if hours > 0:
            return f"{hours}時間{minutes}分" if minutes > 0 else f"{hours}時間"
        else:
            return f"{minutes}分"