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
        self.responses = []
        self.last_response = None

    def set_streamlit_callback(self, callback: Callable[[str], str]):
        """Streamlit用のコールバック関数を設定"""
        self.streamlit_input_callback = callback

    def _load_prompts(self) -> Dict[str, str]:
        """プロンプトファイルを読み込む"""
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts.json")
        with open(prompt_path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def run(
        self,
        persona: Persona,
        policy: TeachingPolicy,
        conversation_history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """エージェントのメイン実行メソッド"""
        if conversation_history:
            self.conversation_history.messages = conversation_history

        try:
            # 質問の生成
            questions = await self._generate_initial_questions(persona, policy)
            collected_responses = []
            
            # モードに応じた処理
            if self.mode == "cli":
                for question in questions:
                    if self.current_turn >= self.max_turns:
                        break
                    
                    response = await self._get_user_response(question)
                    if response:
                        self.conversation_history.messages.append(("assistant", question))
                        self.conversation_history.messages.append(("user", response))
                        collected_responses.append(response)
                        self.current_turn += 1

            elif self.mode == "streamlit":
                for question in questions:
                    if self.current_turn >= self.max_turns:
                        break
                    
                    response = None
                    if self.streamlit_input_callback:
                        response = self.streamlit_input_callback(question)
                        
                    if response:
                        self.conversation_history.messages.append(("assistant", question))
                        self.conversation_history.messages.append(("user", response))
                        collected_responses.append(response)
                        self.current_turn += 1

            elif self.mode == "mock":
                # モックモードの処理
                for question in questions:
                    mock_response = f"【Mock Response】for: {question}"
                    self.conversation_history.messages.append(("assistant", question))
                    self.conversation_history.messages.append(("user", mock_response))
                    collected_responses.append(mock_response)
                    self.current_turn += 1

            # 会話から洞察を抽出
            insights = await self._extract_insights()

            # 最後の応答を保存
            self.responses = collected_responses
            self.last_response = collected_responses[-1] if collected_responses else None

            return {
                "conversation_history": self.conversation_history.messages,
                "interactive_insights": insights,
                "last_response": self.last_response,
                "responses": self.responses
            }

        except Exception as e:
            self.logger.log_error_details(error=e, agent=self.agent_name)
            return {
                "conversation_history": self.conversation_history.messages,
                "interactive_insights": [],
                "last_response": None,
                "responses": []
            }

    async def _generate_initial_questions(self, persona: Any, policy: Any) -> List[str]:
        """初期質問を生成"""
        # 辞書に変換する処理を追加
        persona_dict = persona.dict() if hasattr(persona, 'dict') else persona
        policy_dict = policy.dict() if hasattr(policy, 'dict') else policy
        
        prompt = self.prompts["initial_questions_prompt"].format(
            persona=json.dumps(persona_dict, ensure_ascii=False),
            policy=json.dumps(policy_dict, ensure_ascii=False)
        )
        response = await self.llm.ainvoke(prompt)
        return self._parse_questions(response.content)

    async def _get_user_response(self, question: str) -> str:
        """各モードに応じたユーザー回答の取得"""
        if self.mode == "cli":
            print(f"\nAssistant: {question}")
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                response = await loop.run_in_executor(pool, input, "You> ")
            return response.strip()
        
        elif self.mode == "streamlit":
            if self.streamlit_input_callback:
                return self.streamlit_input_callback(question)
            return ""
            
        elif self.mode == "mock":
            return f"【Mock Response】for: {question}"
            
        return ""

    async def _extract_insights(self) -> List[str]:
        """会話から洞察を抽出"""
        prompt = self.prompts["insight_extraction_prompt"].format(
            conversation_history=json.dumps(
                self.conversation_history.dict(),
                ensure_ascii=False
            )
        )
        response = await self.llm.ainvoke(prompt)
        return self._parse_insights(response.content)

    def _parse_questions(self, content: str) -> List[str]:
        """質問文字列をリストに分解"""
        lines = content.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]

    def _parse_insights(self, content: str) -> List[str]:
        """洞察文字列をリストに分解"""
        lines = content.strip().split("\n")
        return [line.strip() for line in lines if line.strip()]