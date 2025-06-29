# utils/validators.py
from typing import Optional, Dict, Any
import os

class DataValidator:
    @staticmethod
    def validate_video_file(file_path: str) -> bool:
        """動画ファイルの妥当性を検証"""
        if not os.path.exists(file_path):
            return False
            
        # 拡張子チェック
        valid_extensions = {'.mp4', '.mov', '.avi'}
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in valid_extensions:
            return False
            
        # ファイルサイズチェック
        if os.path.getsize(file_path) == 0:
            return False
            
        return True

    @staticmethod
    def validate_json_data(data: Dict[str, Any]) -> Optional[str]:
        """JSONデータの妥当性を検証し、エラーメッセージを返す"""
        # 最小限のフィールドの存在チェックのみ
        if not all(key in data for key in ["basic_info", "coaching_policy"]):
            return "Missing required fields: basic_info or coaching_policy"
            
        return None