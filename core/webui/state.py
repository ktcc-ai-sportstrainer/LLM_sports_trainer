from typing import Dict, Any, Optional
from datetime import datetime
from core.base.state import BaseState

class WebUIState(BaseState):
    """WebUI用の状態管理クラス"""
    def __init__(self, initial_state: Dict[str, Any]):
        self.state = initial_state
        self.history = []
        self.processing_status = {
            "current_step": None,
            "progress": 0,
            "status": "initialized",
            "error": None
        }
        self.display_paths = {
            "user_video": None,
            "ideal_video": None,
            "visualization": None
        }

    def update(self, updates: Dict[str, Any]) -> None:
        """状態を更新"""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "state": self.state.copy(),
            "processing_status": self.processing_status.copy()
        })
        self.state.update(updates)

    def get_state(self) -> Dict[str, Any]:
        """現在の状態を取得"""
        return {
            "main_state": self.state.copy(),
            "processing_status": self.processing_status.copy(),
            "display_paths": self.display_paths.copy()
        }

    def update_processing_status(
        self,
        step: Optional[str] = None,
        progress: Optional[int] = None,
        status: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """処理状態を更新"""
        if step is not None:
            self.processing_status["current_step"] = step
        if progress is not None:
            self.processing_status["progress"] = progress
        if status is not None:
            self.processing_status["status"] = status
        if error is not None:
            self.processing_status["error"] = error

    def set_display_path(self, video_type: str, path: str) -> None:
        """動画表示パスを設定"""
        if video_type not in self.display_paths:
            raise ValueError(f"Invalid video type: {video_type}")
        self.display_paths[video_type] = path

    def get_display_path(self, video_type: str) -> Optional[str]:
        """動画表示パスを取得"""
        return self.display_paths.get(video_type)

    def clear_display_paths(self) -> None:
        """全ての表示パスをクリア"""
        self.display_paths = {
            "user_video": None,
            "ideal_video": None,
            "visualization": None
        }

    def get_progress(self) -> Tuple[str, int, str]:
        """現在の進捗状況を取得"""
        return (
            self.processing_status["current_step"],
            self.processing_status["progress"],
            self.processing_status["status"]
        )

    def is_interactive_mode(self) -> bool:
        """インタラクティブモードが有効かどうかを確認"""
        return self.state.get("interactive_mode", True)

    def set_interactive_mode(self, enabled: bool) -> None:
        """インタラクティブモードの設定"""
        self.state["interactive_mode"] = enabled

    def get_errors(self) -> List[Dict[str, Any]]:
        """エラー履歴を取得"""
        return self.state.get("errors", [])

    def add_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """エラーを追加"""
        if "errors" not in self.state:
            self.state["errors"] = []
        
        self.state["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        })

    def clear_errors(self) -> None:
        """エラー履歴をクリア"""
        self.state["errors"] = []

    def get_last_error(self) -> Optional[Dict[str, Any]]:
        """最新のエラーを取得"""
        errors = self.get_errors()
        return errors[-1] if errors else None

    def get_processing_history(self) -> List[Dict[str, Any]]:
        """処理履歴を取得"""
        return [
            {
                "timestamp": h["timestamp"],
                "step": h["processing_status"]["current_step"],
                "status": h["processing_status"]["status"]
            }
            for h in self.history
        ]

    def reset(self) -> None:
        """状態を初期化"""
        self.state = {
            "interactive_mode": True,
            "errors": []
        }
        self.processing_status = {
            "current_step": None,
            "progress": 0,
            "status": "initialized",
            "error": None
        }
        self.clear_display_paths()
        self.history = []

    def to_dict(self) -> Dict[str, Any]:
        """全ての状態をdict形式で取得"""
        return {
            "state": self.state,
            "processing_status": self.processing_status,
            "display_paths": self.display_paths,
            "history": self.history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebUIState':
        """dict形式から状態を復元"""
        instance = cls(data["state"])
        instance.processing_status = data["processing_status"]
        instance.display_paths = data["display_paths"]
        instance.history = data["history"]
        return instance