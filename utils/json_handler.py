import json
from typing import Any, Dict, Type, TypeVar
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class JSONHandler:
    @staticmethod
    def load_json(file_path: str) -> Dict[str, Any]:
        """JSONファイルを読み込む"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str):
        """データをJSONファイルとして保存"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_model(file_path: str, model_class: Type[T]) -> T:
        """JSONファイルを指定されたPydanticモデルとして読み込む"""
        data = JSONHandler.load_json(file_path)
        return model_class(**data)

    @staticmethod
    def save_model(model: BaseModel, file_path: str):
        """PydanticモデルをJSONとして保存"""
        JSONHandler.save_json(model.dict(), file_path)
        
    def validate_agent_output(self, agent_name: str, output: Dict[str, Any]) -> bool:
        validation_schema = {
            "goal_setting": self._validate_goals_output,
            "search": self._validate_search_output,
            "plan": self._validate_plan_output
        }
        return validation_schema.get(agent_name, lambda x: True)(output)