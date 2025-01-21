from typing import Any, Dict, List
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base import BaseAgent
from models.input.persona import Persona
from models.input.policy import TeachingPolicy
from models.internal.goal import Goal

class GoalSettingAgent(BaseAgent):
    """
    選手の情報、指導方針、分析結果を基に適切な目標を設定するエージェント
    """

    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        
        # プロンプトの読み込み
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
            
        self.goal_prompt = ChatPromptTemplate.from_template(prompts["goal_prompt"])
        self.validation_prompt = ChatPromptTemplate.from_template(prompts["validation_prompt"])
        self.metrics_prompt = ChatPromptTemplate.from_template(prompts["metrics_prompt"])

    async def run(
        self,
        persona: Persona,
        policy: TeachingPolicy,
        conversation_insights: List[str],
        motion_analysis: str
    ) -> Dict[str, Any]:
        """
        目標設定の実行
        """
        # 1. 初期目標の生成
        initial_goals = await self._generate_initial_goals(
            persona, policy, conversation_insights, motion_analysis
        )
        
        # 2. 目標の検証と調整
        validated_goals = await self._validate_goals(initial_goals, persona, policy)
        
        # 3. 達成指標の設定
        metrics = await self._set_metrics(validated_goals)
        
        # 4. Goal オブジェクトの生成
        final_goal = Goal(
            primary_goal=validated_goals["primary_goal"],
            sub_goals=validated_goals["sub_goals"],
            metrics=metrics
        )
        
        return self.create_output(
            output_type="goal_setting",
            content=final_goal.dict()
        ).dict()

    async def _generate_initial_goals(
        self,
        persona: Persona,
        policy: TeachingPolicy,
        conversation_insights: List[str],
        motion_analysis: str
    ) -> Dict[str, Any]:
        """初期目標の生成"""
        response = await self.llm.ainvoke(
            self.goal_prompt.format(
                persona=persona.dict(),
                policy=policy.dict(),
                insights="\n".join(conversation_insights),
                analysis=motion_analysis
            )
        )
        
        goals_dict = json.loads(response.content)
        return goals_dict

    async def _validate_goals(
        self,
        goals: Dict[str, Any],
        persona: Persona,
        policy: TeachingPolicy
    ) -> Dict[str, Any]:
        """目標の妥当性検証と調整"""
        response = await self.llm.ainvoke(
            self.validation_prompt.format(
                goals=json.dumps(goals, ensure_ascii=False),
                persona=persona.dict(),
                policy=policy.dict()
            )
        )
        
        validated_goals = json.loads(response.content)
        return validated_goals

    async def _set_metrics(self, goals: Dict[str, Any]) -> List[str]:
        """目標達成の指標を設定"""
        response = await self.llm.ainvoke(
            self.metrics_prompt.format(
                primary_goal=goals["primary_goal"],
                sub_goals=goals["sub_goals"]
            )
        )
        
        metrics = json.loads(response.content)
        return metrics["metrics"]

    def _validate_metrics(self, metrics: List[str]) -> bool:
        """指標の妥当性をチェック"""
        for metric in metrics:
            if not isinstance(metric, str) or len(metric.strip()) == 0:
                return False
                
            # 測定可能性の簡易チェック
            if not any(keyword in metric.lower() for keyword in [
                "回数", "時間", "速度", "角度", "成功率", "得点", "ヒット率"
            ]):
                return False
        
        return True