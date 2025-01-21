from pydantic import BaseModel, Field
from typing import List, Tuple


class ConversationHistory(BaseModel):
    messages: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="対話履歴のリスト（質問、回答のタプル）"
    )
    key_insights: List[str] = Field(
        default_factory=list,
        description="対話から得られた重要な洞察"
    )