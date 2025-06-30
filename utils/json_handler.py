
import json
from typing import Any, Dict, Type, TypeVar

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