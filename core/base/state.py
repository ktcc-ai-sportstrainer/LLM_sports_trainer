from typing import Dict, Any, List, Optional, TypeVar
from datetime import datetime
from abc import ABC, abstractmethod

class BaseState(ABC):
    """基本状態クラス"""
    @abstractmethod
    def update(self, updates: Dict[str, Any]) -> None:
        """状態を更新"""
        pass

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """現在の状態を取得"""
        pass

class SystemState(BaseState):
    """システム全体の状態を管理する基本クラス"""
    def __init__(self, initial_state: Dict[str, Any]):
        self.state = initial_state
        self.history: List[Dict[str, Any]] = []
        
    def update(self, updates: Dict[str, Any]) -> None:
        """状態を更新し、履歴を保存"""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "state": self.state.copy()
        })
        self.state.update(updates)

    def get_state(self) -> Dict[str, Any]:
        """現在の状態を取得"""
        return self.state.copy()

    def get_history(self) -> List[Dict[str, Any]]:
        """状態の履歴を取得"""
        return self.history.copy()

def create_initial_state(
    persona_data: Dict[str, Any],
    policy_data: Dict[str, Any],
    user_video_path: Optional[str] = None,
    ideal_video_path: Optional[str] = None,
    user_pose_json: Optional[str] = None,
    ideal_pose_json: Optional[str] = None
) -> Dict[str, Any]:
    """初期状態を生成"""
    return {
        # 基本情報
        "persona_data": persona_data,
        "policy_data": policy_data,
        "user_video_path": user_video_path,
        "ideal_video_path": ideal_video_path,
        "user_pose_json": user_pose_json,
        "ideal_pose_json": ideal_pose_json,

        # エージェントの出力（初期状態）
        "conversation": [],
        "motion_analysis": "",
        "goals": "",
        "search_queries": "",
        "search_results": "",
        "plan": "",
        "summary": "",

        # 実行状態
        "status": "initialized",
        "errors": [],
        "last_agent": "",
        "processing_step": None,
        "current_progress": 0
    }