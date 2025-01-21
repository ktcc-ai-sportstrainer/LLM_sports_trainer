from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from models.output.agent_output import AgentOutput


class BaseAgent(ABC):
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.agent_name = self.__class__.__name__

    @abstractmethod
    async def run(self, *args, **kwargs) -> Dict[str, Any]:
        """エージェントの主要な処理を実行"""
        pass

    def create_output(self, output_type: str, content: Dict[str, Any]) -> AgentOutput:
        """エージェント出力を生成"""
        return AgentOutput(
            agent_name=self.agent_name,
            output_type=output_type,
            content=content,
            timestamp=datetime.now().isoformat()
        )

    def _validate_input(self, *args, **kwargs) -> bool:
        """入力の妥当性を検証"""
        return True