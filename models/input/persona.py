from pydantic import BaseModel, Field


class Persona(BaseModel):
    age: int = Field(..., description="選手の年齢")
    experience: str = Field(..., description="野球経験年数")
    level: str = Field(..., description="現在のレベル")
    height: float = Field(..., description="身長(cm)")
    weight: float = Field(..., description="体重(kg)")
    position: str = Field(..., description="ポジション")
    batting_style: str = Field(..., description="打席（右打ち/左打ち/両打ち）")