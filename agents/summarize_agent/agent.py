from typing import Any, Dict
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base import BaseAgent

class SummarizeAgent(BaseAgent):
    """
    各エージェントの出力を統合し、最終的なコーチングレポートを生成するエージェント
    """

    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        
        # プロンプトの読み込み
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
            
        self.summary_prompt = ChatPromptTemplate.from_template(prompts["summary_prompt"])
        self.action_plan_prompt = ChatPromptTemplate.from_template(prompts["action_plan_prompt"])
        self.feedback_prompt = ChatPromptTemplate.from_template(prompts["feedback_prompt"])

    async def run(
        self,
        analysis: Dict[str, Any],
        goal: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        分析結果、目標設定、練習計画を統合して最終レポートを生成
        """
        # 1. 全体サマリーの生成
        summary = await self._generate_summary(analysis, goal, plan)
        
        # 2. アクションプランの生成
        action_plan = await self._generate_action_plan(goal, plan)
        
        # 3. フィードバックポイントの生成
        feedback = await self._generate_feedback(analysis, goal)
        
        # 4. 最終レポートの構造化
        final_report = {
            "summary": summary,
            "action_plan": action_plan,
            "feedback_points": feedback
        }
        
        return self.create_output(
            output_type="final_summary",
            content=final_report
        ).dict()

    async def _generate_summary(
        self,
        analysis: Dict[str, Any],
        goal: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> str:
        """全体サマリーの生成"""
        response = await self.llm.ainvoke(
            self.summary_prompt.format(
                analysis=json.dumps(analysis, ensure_ascii=False),
                goal=json.dumps(goal, ensure_ascii=False),
                plan=json.dumps(plan, ensure_ascii=False)
            )
        )
        
        summary_dict = json.loads(response.content)
        
        # キーポイントを箇条書きテキストに変換
        summary_points = []
        for category, points in summary_dict.items():
            summary_points.append(f"【{category}】")
            for point in points:
                summary_points.append(f"・{point}")
            summary_points.append("")  # 空行を追加
        
        return "\n".join(summary_points)

    async def _generate_action_plan(
        self,
        goal: Dict[str, Any],
        plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """具体的なアクションプランの生成"""
        response = await self.llm.ainvoke(
            self.action_plan_prompt.format(
                goal=json.dumps(goal, ensure_ascii=False),
                plan=json.dumps(plan, ensure_ascii=False)
            )
        )
        
        action_plan = json.loads(response.content)
        
        # アクションプランの妥当性チェック
        if self._validate_action_plan(action_plan):
            return action_plan
        else:
            # 不正な形式の場合、基本構造を返す
            return {
                "immediate_actions": [],
                "weekly_schedule": {},
                "milestones": []
            }

    async def _generate_feedback(
        self,
        analysis: Dict[str, Any],
        goal: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """フィードバックポイントの生成"""
        response = await self.llm.ainvoke(
            self.feedback_prompt.format(
                analysis=json.dumps(analysis, ensure_ascii=False),
                goal=json.dumps(goal, ensure_ascii=False)
            )
        )
        
        feedback = json.loads(response.content)
        
        # フィードバックの構造化と優先順位付け
        return self._structure_feedback(feedback)

    def _validate_action_plan(self, plan: Dict[str, Any]) -> bool:
        """アクションプランの妥当性チェック"""
        required_keys = {"immediate_actions", "weekly_schedule", "milestones"}
        if not all(key in plan for key in required_keys):
            return False
            
        # immediate_actionsのチェック
        if not isinstance(plan["immediate_actions"], list):
            return False
            
        # weekly_scheduleのチェック
        if not isinstance(plan["weekly_schedule"], dict):
            return False
            
        # milestonesのチェック
        if not isinstance(plan["milestones"], list):
            return False
        
        return True

    def _structure_feedback(
        self,
        feedback: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """フィードバックの構造化"""
        structured_feedback = []
        
        # 優先度に基づいて並べ替え
        for category in ["critical", "important", "nice_to_have"]:
            if category in feedback:
                for point in feedback[category]:
                    structured_feedback.append({
                        "priority": category,
                        "point": point["content"],
                        "reason": point.get("reason", ""),
                        "suggestion": point.get("suggestion", "")
                    })
        
        return structured_feedback