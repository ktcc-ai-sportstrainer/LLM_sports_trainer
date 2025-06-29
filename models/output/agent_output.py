from datetime import datetime
from typing import Dict, Any

class AgentOutput:
    """
    各エージェントからの出力を統一的な形式で保持したい場合のクラス
    """
    def __init__(self, agent_name: str, output_type: str, content: Dict[str, Any], timestamp: str = None):
        self.agent_name = agent_name
        self.output_type = output_type
        self.content = content
        self.timestamp = timestamp if timestamp else datetime.now().isoformat()

    def dict(self) -> Dict[str, Any]:
        """
        pydanticライクに dict() を返すメソッド
        """
        return {
            "agent_name": self.agent_name,
            "output_type": self.output_type,
            "content": self.content,
            "timestamp": self.timestamp
        }
