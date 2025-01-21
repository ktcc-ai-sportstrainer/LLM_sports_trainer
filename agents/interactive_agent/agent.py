from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json
import os

from agents.base import BaseAgent
from models.internal.conversation import ConversationHistory
from models.input.persona import Persona
from models.input.policy import TeachingPolicy

class InteractiveAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        
        # プロンプトの読み込み
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)
        
        self.main_prompt = ChatPromptTemplate.from_template(prompts["main_prompt"])
        self.follow_up_prompt = ChatPromptTemplate.from_template(prompts["follow_up_prompt"])
        self.insight_prompt = ChatPromptTemplate.from_template(prompts["insight_prompt"])

    async def run(
        self,
        persona: Persona,
        policy: TeachingPolicy,
        conversation_history: ConversationHistory,
    ) -> Dict[str, Any]:
        """質問生成と対話の実行"""
        # 最初の質問セットを生成
        questions = await self._generate_questions(persona, policy, conversation_history)
        
        # フォローアップ質問の生成（必要に応じて）
        if conversation_history.messages:
            follow_up_questions = await self._generate_follow_up(
                persona, policy, conversation_history, questions
            )
            questions.extend(follow_up_questions)
        
        # 対話から得られた洞察を抽出
        insights = await self._extract_insights(conversation_history)
        
        return self.create_output(
            output_type="interactive_results",
            content={
                "questions": questions,
                "insights": insights
            }
        ).dict()

    async def _generate_questions(
        self,
        persona: Persona,
        policy: TeachingPolicy,
        conversation_history: ConversationHistory
    ) -> List[str]:
        """初期質問の生成"""
        response = await self.llm.ainvoke(
            self.main_prompt.format(
                persona=persona.dict(),
                policy=policy.dict(),
                conversation_history=self._format_history(conversation_history)
            )
        )
        
        # 応答から質問を抽出
        questions = [q.strip() for q in response.content.split("\n") if q.strip()]
        return questions[:3]  # 最大3問に制限

    async def _generate_follow_up(
        self,
        persona: Persona,
        policy: TeachingPolicy,
        conversation_history: ConversationHistory,
        previous_questions: List[str]
    ) -> List[str]:
        """フォローアップ質問の生成"""
        response = await self.llm.ainvoke(
            self.follow_up_prompt.format(
                persona=persona.dict(),
                policy=policy.dict(),
                conversation_history=self._format_history(conversation_history),
                previous_questions="\n".join(previous_questions)
            )
        )
        
        questions = [q.strip() for q in response.content.split("\n") if q.strip()]
        return questions[:2]  # 最大2問に制限

    async def _extract_insights(
        self,
        conversation_history: ConversationHistory
    ) -> List[str]:
        """対話から重要な洞察を抽出"""
        if not conversation_history.messages:
            return []
            
        response = await self.llm.ainvoke(
            self.insight_prompt.format(
                conversation_history=self._format_history(conversation_history)
            )
        )
        
        insights = [i.strip() for i in response.content.split("\n") if i.strip()]
        return insights

    def _format_history(self, conv_history: ConversationHistory) -> str:
        """会話履歴のフォーマット"""
        formatted = []
        for role, message in conv_history.messages:
            formatted.append(f"{role}: {message}")
        return "\n".join(formatted)