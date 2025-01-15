from pydantic import BaseModel, Field
from models.role import Role

class Task(BaseModel):
    description: str = Field(..., description="タスクの説明")
    role: Role = Field(default=None, description="タスクに割り当てられた役割")

class TaskWithRoles(BaseModel):
    tasks: list[Task] = Field(...,description="役割が割り当てられたタスクのリスト")