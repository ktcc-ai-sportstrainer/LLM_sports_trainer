from typing import Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agents.base import BaseAgent
from models.input.persona import Persona
from models.input.policy import TeachingPolicy
from models.internal.goal import Goal
from models.output.agent_output import AgentOutput

class GoalSettingAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        self.prompt = ChatPromptTemplate.from_template(
            """あなたは経験豊富な野球コーチです。
            以下の情報を基に、選手の目標設定を行ってください。
            
            選手情報:
            {persona}
            
            指導方針:
            {policy}
            
            対話から得られた洞察:
            {conversation_insights}
            
            動作解析結果:
            {motion_analysis}
            
            以下の要素を考慮して目標を設定してください：
            1. 技術的な課題の改善
            2. 選手の希望や目標
            3. 現実的な達成可能性
            4. 測定可能な指標
            5. 段階的な成長プロセス
            
            出力形式：
            - 主目標（1つ）
            - サブ目標（2-3個）
            - 達成度を測る指標（各目標に対して）
            """
        )

    async def run(
        self,
        persona: Persona,
        policy: TeachingPolicy,
        conversation_insights: list[str],
        motion_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        # 目標設定の生成
        response = await self.llm.ainvoke(
            self.prompt.format(
                persona=persona.dict(),
                policy=policy.dict(),
                conversation_insights="\n".join(conversation_insights),
                motion_analysis=motion_analysis
            )
        )
        
        # レスポンスをパース
        goals = await self._parse_goals(response.content)
        
        # 目標の検証
        validated_goals = await self._validate_goals(goals)
        
        return self.create_output(
            output_type="goal_setting",
            content=validated_goals.dict()
        ).dict()

    async def _parse_goals(self, response: str) -> Goal:
        parse_prompt = ChatPromptTemplate.from_template(
            """以下の目標設定を、主目標、サブ目標、測定指標に分類してください：
            
            {response}
            
            出力は以下の形式で行ってください：
            主目標：（一つだけ記載）
            サブ目標：（箇条書きで記載）
            測定指標：（各目標に対応する指標を箇条書きで記載）
            """
        )
        
        parsed = await self.llm.ainvoke(
            parse_prompt.format(response=response)
        )
        
        # パース結果を構造化
        lines = parsed.content.split("\n")
        primary_goal = ""
        sub_goals = []
        metrics = []
        
        current_section = None
        for line in lines:
            if "主目標：" in line:
                current_section = "primary"
                primary_goal = line.split("：")[1].strip()
            elif "サブ目標：" in line:
                current_section = "sub"
            elif "測定指標：" in line:
                current_section = "metrics"
            elif line.strip():
                if current_section == "sub":
                    sub_goals.append(line.strip())
                elif current_section == "metrics":
                    metrics.append(line.strip())
        
        return Goal(
            primary_goal=primary_goal,
            sub_goals=sub_goals,
            metrics=metrics
        )

    async def _validate_goals(self, goals: Goal) -> Goal:
        validation_prompt = ChatPromptTemplate.from_template(
            """以下の目標設定が適切かどうか確認してください：
            
            主目標：{primary_goal}
            サブ目標：
            {sub_goals}
            測定指標：
            {metrics}
            
            確認項目：
            1. 具体的で明確か
            2. 測定可能か
            3. 達成可能か
            4. 現実的か
            5. 期限は適切か
            
            問題がある場合は修正した目標を、問題がない場合はそのまま目標を出力してください。
            """
        )
        
        validated = await self.llm.ainvoke(
            validation_prompt.format(
                primary_goal=goals.primary_goal,
                sub_goals="\n".join(goals.sub_goals),
                metrics="\n".join(goals.metrics)
            )
        )
        
        # 検証結果を再度パース
        return await self._parse_goals(validated.content)