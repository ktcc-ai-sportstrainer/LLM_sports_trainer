from pydantic import BaseModel, Field


# ... (Goalクラスのコード)
class Goal(BaseModel):
    description: str = Field(..., description="目標の設定")

    @property
    def text(self) -> str:
        return f"{self.description}"
