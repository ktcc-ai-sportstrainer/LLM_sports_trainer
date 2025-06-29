from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, TypeVar

from langchain_openai import ChatOpenAI
from models.output.agent_output import AgentOutput
from core.base.logger import SystemLogger


class BaseAgent(ABC):
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.agent_name = self.__class__.__name__
        self.logger = SystemLogger()

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
