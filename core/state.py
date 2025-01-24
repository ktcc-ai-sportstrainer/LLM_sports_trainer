from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
from pydantic import BaseModel

class AgentState(TypedDict):
    persona_data: Dict[str, Any]
    policy_data: Dict[str, Any]
    user_video_path: Optional[str]
    ideal_video_path: Optional[str]
    conversation: List[Dict[str, str]]
    motion_analysis: str # ModelingAgentの出力形式変更に対応
    goals: str
    search_queries: str
    search_results: str
    plan: str
    summary: str

def create_initial_state(
    persona_data: Dict[str, Any],
    policy_data: Dict[str, Any],
    user_video_path: Optional[str],
    ideal_video_path: Optional[str]
) -> AgentState:
    """初期状態を生成"""
    return {
        # 基本情報
        "persona_data": persona_data,
        "policy_data": policy_data,
        "user_video_path": user_video_path,
        "ideal_video_path": ideal_video_path,

        # エージェントの出力（初期状態）
        "conversation": [],
        "motion_analysis": "", # 文字列に変更
        "goals": "",
        "search_queries": "",
        "search_results": "",
        "plan": "",
        "summary": "",

        # 実行状態
        "status": "started",
        "errors": [],
        "last_agent": ""
    }

class StateManager:
    """状態の更新と管理を行うクラス"""
    
    def __init__(self, initial_state: AgentState):
        self.state = initial_state
        self.history: List[Dict[str, Any]] = []
        
    def update_state(self, agent_name: str, updates: Dict[str, Any], error: Optional[Exception] = None) -> AgentState:
        """状態を更新"""
        # 現在の状態を履歴に保存
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "state": self.state.copy()
        })
        
        # 新しい状態を作成
        new_state = {
            **self.state,
            **updates,
            "last_agent": agent_name
        }
        
        # エラーが発生した場合
        if error:
            new_state["status"] = "error"
            new_state["errors"] = new_state.get("errors", []) + [{
                "agent": agent_name,
                "error": str(error),
                "timestamp": datetime.now().isoformat()
            }]
        
        self.state = new_state
        return self.state
    
    def get_agent_input(self, agent_name: str) -> Dict[str, Any]:
        """特定のエージェントに必要な入力を取得"""
        if agent_name == "interactive":
            return {
                "persona": self.state["persona_data"],
                "policy": self.state["policy_data"]
            }
        elif agent_name == "modeling":
            return {
                "user_video_path": self.state["user_video_path"],
                "ideal_video_path": self.state["ideal_video_path"],
                "user_pose_json": self.state.get("user_pose_json"),
                "ideal_pose_json": self.state.get("ideal_pose_json")
            }
        elif agent_name == "goal_setting":
            return {
                "persona": self.state["persona_data"],
                "policy": self.state["policy_data"],
                "conversation_insights": self.state["conversation"],
                "motion_analysis": self.state["motion_analysis"]
            }
        elif agent_name == "search":
            return {
                "search_queries": self.state["search_queries"]
            }
        elif agent_name == "plan":
            return {
                "goal": self.state["goals"],
                "motion_analysis": self.state["motion_analysis"],
            }
        elif agent_name == "summarize":
            return {
                "analysis": self.state["motion_analysis"],
                "goal": self.state["goals"],
                "plan": self.state["plan"]
            }
        else:
            return {}

    def get_execution_status(self) -> Dict[str, Any]:
        """実行状態の概要を取得"""
        return {
            "status": self.state["status"],
            "last_agent": self.state["last_agent"],
            "error_count": len(self.state["errors"]),
            "completed_steps": [h["agent"] for h in self.history]
        }