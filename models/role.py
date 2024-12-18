from pydantic import BaseModel, Field


class Role(BaseModel):
    name: str = Field(..., description="役割の名前")
    description: str = Field(..., description="役割の詳細な説明")
    key_skills: list[str] = Field(..., description="この役割に必要な主要スキルや属性")
