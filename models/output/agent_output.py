from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

class AgentMetrics(BaseModel):
    execution_time: float = Field(..., description="実行時間（秒）")
    memory_usage: Optional[float] = Field(default=None, description="メモリ使用量（MB）")
    api_calls: Optional[int] = Field(default=None, description="API呼び出し回数")
    tokens_used: Optional[Dict[str, int]] = Field(default=None, description="使用トークン数（入力/出力）")

class AgentError(BaseModel):
    error_type: str = Field(..., description="エラーの種類")
    message: str = Field(..., description="エラーメッセージ")
    traceback: Optional[str] = Field(default=None, description="エラーのトレースバック")
    timestamp: datetime = Field(default_factory=datetime.now)

class AgentOutput(BaseModel):
    agent_name: str = Field(..., description="エージェントの名前")
    output_type: str = Field(..., description="出力の種類")
    content: Dict[str, Any] = Field(..., description="エージェントの出力内容")
    timestamp: datetime = Field(default_factory=datetime.now, description="処理時刻")
    metrics: Optional[AgentMetrics] = Field(default=None, description="実行メトリクス")
    errors: Optional[List[AgentError]] = Field(default=None, description="発生したエラー")
    dependencies: Optional[Dict[str, str]] = Field(default=None, description="依存した他のエージェントの出力")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def add_error(self, error: Exception) -> None:
        """エラー情報の追加"""
        if self.errors is None:
            self.errors = []
            
        self.errors.append(AgentError(
            error_type=type(error).__name__,
            message=str(error),
            traceback=''.join(traceback.format_tb(error.__traceback__))
        ))

    def update_metrics(self, 
                      execution_time: float,
                      memory_usage: Optional[float] = None,
                      api_calls: Optional[int] = None,
                      tokens_used: Optional[Dict[str, int]] = None) -> None:
        """メトリクス情報の更新"""
        self.metrics = AgentMetrics(
            execution_time=execution_time,
            memory_usage=memory_usage,
            api_calls=api_calls,
            tokens_used=tokens_used
        )

    def add_dependency(self, agent_name: str, output_id: str) -> None:
        """依存関係の追加"""
        if self.dependencies is None:
            self.dependencies = {}
            
        self.dependencies[agent_name] = output_id