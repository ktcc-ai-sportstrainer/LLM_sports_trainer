from typing import Any, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agents.base import BaseAgent
from models.internal.conversation import ConversationHistory
from models.input.persona import Persona
from models.input.policy import TeachingPolicy

class InteractiveAgent(BaseAgent):
    """
    ペルソナ情報や指導方針を踏まえて、追加でユーザーに質問を行い情報を引き出すエージェント。
    """

    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)

        # 対話のためのプロンプトテンプレート
        self.prompt = ChatPromptTemplate.from_template(
            """あなたは野球のコーチとして、以下の情報(ペルソナと指導方針)を踏まえて、
            追加でユーザーに質問し、より詳細な状況を聞き出してください。

            【選手情報】
            {persona}

            【指導方針】
            {policy}

            これまでの会話履歴:
            {conversation_history}

            質問は2~3個程度に留めてください。
            出力形式：質問の文のみを出力する。
            """
        )

    async def run(
        self,
        persona: Persona,
        policy: TeachingPolicy,
        conversation_history: ConversationHistory,
    ) -> Dict[str, Any]:
        """
        ペルソナと指導方針を受け取り、追加質問を生成して返す。
        """
        response = await self.llm.ainvoke(
            self.prompt.format(
                persona=persona.dict(),
                policy=policy.dict(),
                conversation_history=self._format_history(conversation_history)
            )
        )

        # 質問文をリストとして扱うための簡易パース（改行区切りなど）
        questions = response.content.strip().split("\n")
        questions = [q.strip() for q in questions if q.strip()]

        return self.create_output(
            output_type="interactive_questions",
            content={"questions": questions}
        ).dict()

    def _format_history(self, conv_history: ConversationHistory) -> str:
        """
        ConversationHistory(messages: List[Tuple[str, str]]] などを文字列化
        """
        lines = []
        for role, msg in conv_history.messages:
            lines.append(f"{role}: {msg}")
        return "\n".join(lines)
