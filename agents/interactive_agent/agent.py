from typing import Dict, Any, List, Optional
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
        self.current_turn = 0
        self.max_turns = 3

        self.conversation_history = ConversationHistory()
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> Dict[str, str]:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def run(
        self,
        persona: Persona,
        policy: TeachingPolicy,
    ) -> Dict[str, Any]:
        try:
            # 初期質問の生成
            initial_questions = await self._generate_initial_questions(persona, policy)
            
            # 対話の実行
            for question in initial_questions:
                response = await self._get_user_response(question)
                self.conversation_history.messages.append(("assistant", question))
                self.conversation_history.messages.append(("user", response))
                
                # フォローアップ質問の生成と実行
                if self.current_turn < self.max_turns:
                    follow_up = await self._generate_follow_up(response, persona, policy)
                    if follow_up:
                        follow_up_response = await self._get_user_response(follow_up)
                        self.conversation_history.messages.append(("assistant", follow_up))
                        self.conversation_history.messages.append(("user", follow_up_response))
                        
            # 会話からの洞察を抽出
            insights = await self._extract_insights()
            
            return {
                "conversation_history": self.conversation_history.dict(),
                "insights": insights
            }

        except Exception as e:
            self.logger.error(f"Error in InteractiveAgent: {str(e)}")
            return {}

    async def _generate_initial_questions(
        self,
        persona: Persona,
        policy: TeachingPolicy,
    ) -> List[str]:
        prompt = self.prompts["initial_questions"].format(
            persona=persona.dict(),
            policy=policy.dict()
        )
        response = await self.llm.ainvoke(prompt)
        questions = self._parse_questions(response.content)
        return questions[:3]  # 最大3つの質問を返す

    async def _generate_follow_up(
        self,
        last_response: str,
        persona: Persona,
        policy: TeachingPolicy,
    ) -> Optional[str]:
        prompt = self.prompts["follow_up"].format(
            last_response=last_response,
            persona=persona.dict(),
            policy=policy.dict(),
            conversation_history=self.conversation_history.dict()
        )
        response = await self.llm.ainvoke(prompt)
        return response.content if response.content.strip() else None

    async def _extract_insights(self) -> List[str]:
        prompt = self.prompts["extract_insights"].format(
            conversation_history=self.conversation_history.dict()
        )
        response = await self.llm.ainvoke(prompt)
        return self._parse_insights(response.content)

    def _parse_questions(self, content: str) -> List[str]:
        return [q.strip() for q in content.split('\n') if q.strip()]

    def _parse_insights(self, content: str) -> List[str]:
        return [i.strip() for i in content.split('\n') if i.strip()]

    async def _get_user_response(self, question: str) -> str:
        """実際のシステムでは、ユーザーからの入力を待つ"""
        # モック実装
        return "モック回答: " + question