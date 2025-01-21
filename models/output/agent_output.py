from pydantic import BaseModel, Field
from typing import Any, Dict

class AgentOutput(BaseModel):
    agent_name: str = Field(..., description="エージェントの名前")
    output_type: str = Field(..., description="出力の種類")
    content: Dict[str, Any] = Field(..., description="エージェントの出力内容")
    timestamp: str = Field(..., description="処理時刻")