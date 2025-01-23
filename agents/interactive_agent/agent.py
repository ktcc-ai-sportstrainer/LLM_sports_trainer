
from typing import Dict, Any, List, Optional, Callable
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from agents.base import BaseAgent
from models.internal.conversation import ConversationHistory
from models.input.persona import Persona
from models.input.policy import TeachingPolicy


class InteractiveAgent(BaseAgent):
    """
    部員と対話し、ペルソナ情報から得られない追加の課題や目標を掘り下げるエージェント。
    対話のやり方を以下で切り替え可能:
      - mode="mock": 質問に対してモック回答を返す
      - mode="cli":  端末上でinput()を使用
      - mode="streamlit": StreamlitのUIを想定
    """

    def __init__(self, llm: ChatOpenAI, mode: str = "mock"):
        super().__init__(llm)
        self.current_turn = 0
        self.max_turns = 3  # 例：3ターン
        self.mode = mode  # "mock" / "cli" / "streamlit"

        self.conversation_history = ConversationHistory()
        self.prompts = self._load_prompts()

        # Streamlitモードの場合のコールバック関数
        self.streamlit_input_callback: Optional[Callable[[str], str]] = None

    def set_streamlit_callback(self, callback: Callable[[str], str]):
        """
        Streamlitアプリなどでユーザー入力を取得するための関数を設定
        callback(question: str) -> user_answer: str
        """
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
        """
        対話を行い、会話ログ + 抽出情報 を返す。
        """
        # 既存の会話履歴があれば反映
        if conversation_history:
            self.conversation_history.messages = conversation_history

        try:
            # 1. 初期質問の生成
            questions = await self._generate_initial_questions(persona, policy)

            # 2. 質疑応答ループ
            for i, question in enumerate(questions):
                if self.current_turn >= self.max_turns:
                    break

                self.conversation_history.messages.append(("assistant", question))
                user_answer = await self._get_user_response(question)
                self.conversation_history.messages.append(("user", user_answer))

                self.current_turn += 1

            # 3. フォローアップ (例)
            if self.current_turn < self.max_turns:
                follow_up_q = await self._generate_follow_up(persona, policy)
                if follow_up_q:
                    self.conversation_history.messages.append(("assistant", follow_up_q))
                    user_answer = await self._get_user_response(follow_up_q)
                    self.conversation_history.messages.append(("user", user_answer))
                    self.current_turn += 1

            # 4. 会話から洞察を抽出
            insights = await self._extract_insights()

            return {
                "conversation_history": self.conversation_history.messages,
                "interactive_insights": insights
            }

        except Exception as e:
            self.logger.log_error_details(error=e, agent=self.agent_name)
            return {}

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
        ユーザー回答を取得する。
        mode に応じて分岐:
          - mock : テスト用固定文言
          - cli  : input() で取得
          - streamlit : set_streamlit_callback() で設定されたコールバック呼び出し
        """
        if self.mode == "mock":
            return f"【Mock Response】for: {question}"

        elif self.mode == "cli":
            # 同期的に input() を呼ぶ例
            # 非同期関数内なので、簡易にはこう書くが実際には run_in_executor() などが必要な場合がある
            ans = input(f"\nAssistant: {question}\nYou> ")
            return ans.strip()

        elif self.mode == "streamlit":
            if self.streamlit_input_callback is not None:
                return self.streamlit_input_callback(question)
            else:
                # コールバックが設定されていない場合のfallback
                return "【No streamlit callback provided】"
        else:
            # デフォルトはmock
            return f"【Unrecognized mode: {self.mode} => Mock Response】"
