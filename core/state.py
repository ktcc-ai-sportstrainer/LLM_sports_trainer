from typing import Dict, Any, List, Optional, TypedDict
from datetime import datetime
from pydantic import BaseModel

class AgentState(TypedDict):
    # 基本情報
    persona_data: Dict[str, Any]
    policy_data: Dict[str, Any]
    user_video_path: str
    ideal_video_path: str
    
    # エージェントの出力
    conversation: List[Dict[str, str]]
    motion_analysis: Dict[str, Any]
    goals: Dict[str, Any]
    plan: Dict[str, Any]
    resources: Dict[str, Any]
    summary: str
    
    # 実行状態
    status: str
    errors: List[Dict[str, Any]]
    last_agent: str

def create_initial_state(
    persona_data: Dict[str, Any],
    policy_data: Dict[str, Any],
    user_video_path: str,
    ideal_video_path: str
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
        "motion_analysis": {},
        "goals": {},
        "plan": {},
        "resources": {},
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
        
    def update_state(self, 
                    agent_name: str, 
                    updates: Dict[str, Any],
                    error: Optional[Exception] = None) -> AgentState:
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
        
        # 妥当性を検証
        if not StateValidator.validate_state(new_state):
            raise ValueError("Invalid state update")
            
        self.state = new_state
        return self.state
    
    def get_agent_input(self, agent_name: str) -> Dict[str, Any]:
        """特定のエージェントに必要な入力を取得"""
        inputs = {
            "interactive": {
                "persona": self.state["persona_data"],
                "policy": self.state["policy_data"],
            },
            "modeling": {
                "user_video_path": self.state["user_video_path"],
                "ideal_video_path": self.state["ideal_video_path"],
            },
            "goal_setting": {
                "persona": self.state["persona_data"],
                "policy": self.state["policy_data"],
                "conversation": self.state["conversation"],
                "motion_analysis": self.state["motion_analysis"],
            },
            # 他のエージェントも同様に設定
        }
        return inputs.get(agent_name, {})

    def get_execution_status(self) -> Dict[str, Any]:
        """実行状態の概要を取得"""
        return {
            "status": self.state["status"],
            "last_agent": self.state["last_agent"],
            "error_count": len(self.state["errors"]),
            "completed_steps": [h["agent"] for h in self.history]
        }