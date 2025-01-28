from typing import Dict, Any, List, Optional, Callable
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

from agents.base import BaseAgent
from models.internal.conversation import ConversationHistory
from models.input.persona import Persona
from models.input.policy import TeachingPolicy
import asyncio
from concurrent.futures import ThreadPoolExecutor

class InteractiveAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI, mode: str = "mock"):
        super().__init__(llm)
        self.current_turn = 0
        self.max_turns = 3
        self.mode = mode
        self.conversation_history = ConversationHistory()
        self.prompts = self._load_prompts()
        self.streamlit_input_callback = None
        self.last_response = None

    def set_streamlit_callback(self, callback: Callable[[str], str]):
        self.streamlit_input_callback = callback

    def _load_prompts(self) -> Dict[str, str]:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def run(
        self,
        persona: Persona,
        policy: TeachingPolicy,
        conversation_history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        if conversation_history:
            self.conversation_history.messages = conversation_history

        try:
            # 質問の生成
            questions = await self._generate_initial_questions(persona, policy)
            
            # Streamlitモードでの処理を改善
            if self.mode == "streamlit":
                for question in questions:
                    if self.current_turn >= self.max_turns:
                        break
                    
                    response = None
                    if self.streamlit_input_callback:
                        response = self.streamlit_input_callback(question)
                        
                    if response:  # 有効な応答がある場合のみ進める
                        self.conversation_history.messages.append(("assistant", question))
                        self.conversation_history.messages.append(("user", response))
                        self.last_response = response  # 応答を保存
                        self.current_turn += 1

            # フォローアップ質問の生成（必要な場合）
            if self.current_turn < self.max_turns:
                follow_up = await self._generate_follow_up(persona, policy)
                if follow_up:
                    if self.mode == "streamlit":
                        response = self.streamlit_input_callback(follow_up)
                        if response:
                            self.conversation_history.messages.append(("assistant", follow_up))
                            self.conversation_history.messages.append(("user", response))
                            self.last_response = response  # 応答を保存
                            self.current_turn += 1

            # 会話から洞察を抽出
            insights = await self._extract_insights()

            return {
                "conversation_history": self.conversation_history.messages,
                "interactive_insights": insights
            }

        except Exception as e:
            self.logger.log_error_details(error=e, agent=self.agent_name)
            # エラー時でも最低限の情報を返す
            return {
                "conversation_history": self.conversation_history.messages,
                "interactive_insights": []
            }

    async def _generate_initial_questions(self, persona: Persona, policy: TeachingPolicy) -> List[str]:
        prompt = self.prompts["initial_questions_prompt"].format(
            persona=json.dumps(persona, ensure_ascii=False),
            policy=json.dumps(policy, ensure_ascii=False)
        )
        response = await self.llm.ainvoke(prompt)
        return self._parse_questions(response.content)

    async def _generate_follow_up(self, persona: Persona, policy: TeachingPolicy) -> Optional[str]:
        prompt = self.prompts["follow_up_prompt"].format(
            persona=json.dumps(persona, ensure_ascii=False),
            policy=json.dumps(policy, ensure_ascii=False),
            conversation_history=self.conversation_history.json()
        )
        response = await self.llm.ainvoke(prompt)
        if response.content.strip():
            return response.content.strip()
        return None

    async def _extract_insights(self) -> List[str]:
        prompt = self.prompts["insight_extraction_prompt"].format(
            conversation_history=self.conversation_history.json()
        )
        response = await self.llm.ainvoke(prompt)
        return self._parse_insights(response.content)

    def _parse_questions(self, content: str) -> List[str]:
        lines = content.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

    def _parse_insights(self, content: str) -> List[str]:
        lines = content.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

    async def _get_user_response(self, question: str) -> str:
        """
        モードに応じたユーザー回答の取得
        """
        if self.mode == "mock":
            return f"【Mock Response】for: {question}"

        elif self.mode == "cli":
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                ans = await loop.run_in_executor(pool, input, f"\nAssistant: {question}\nYou> ")
            return ans.strip()

        elif self.mode == "streamlit":
            if self.streamlit_input_callback is not None:
                return self.streamlit_input_callback(question)
            else:
                return "【No streamlit callback provided】"
        else:
            return f"【Unrecognized mode: {self.mode} => Mock Response】"