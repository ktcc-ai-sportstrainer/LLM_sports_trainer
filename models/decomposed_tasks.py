from pydantic import BaseModel, Field


class DecomposedTasks(BaseModel):
    values: list[str] = Field(
        default_factory=list,
        min_items=3,
        max_items=5,
        description="3～5個に分解されたタスク",
    )
