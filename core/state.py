from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class SystemState(BaseModel):
    """システム全体の状態を管理するクラス"""
    
    # 入力情報
    persona_data: Dict[str, Any]
    policy_data: Dict[str, Any]
    video_path: str
    
    # 各エージェントの処理状態
    interactive_completed: bool = False
    modeling_completed: bool = False
    goal_setting_completed: bool = False
    plan_completed: bool = False
    search_completed: bool = False
    summary_completed: bool = False
    
    # エージェントの出力結果
    conversation_history: List[Dict[str, str]] = []
    motion_analysis: Optional[Dict[str, Any]] = None
    goals: Optional[Dict[str, Any]] = None
    training_plan: Optional[Dict[str, Any]] = None
    search_results: Optional[Dict[str, Any]] = None
    final_summary: Optional[Dict[str, Any]] = None
    
    # エラー状態
    errors: List[Dict[str, Any]] = []
    
    def update_agent_status(self, agent_name: str, completed: bool = True) -> None:
        """エージェントの処理状態を更新"""
        status_field = f"{agent_name.lower()}_completed"
        if hasattr(self, status_field):
            setattr(self, status_field, completed)

    def add_agent_result(self, agent_name: str, result: Dict[str, Any]) -> None:
        """エージェントの処理結果を保存"""
        if agent_name == "interactive":
            self.conversation_history.extend(result.get("conversation", []))
        elif agent_name == "modeling":
            self.motion_analysis = result
        elif agent_name == "goal_setting":
            self.goals = result
        elif agent_name == "plan":
            self.training_plan = result
        elif agent_name == "search":
            self.search_results = result
        elif agent_name == "summary":
            self.final_summary = result

    def add_error(self, agent_name: str, error: Exception) -> None:
        """エラー情報を記録"""
        self.errors.append({
            "agent": agent_name,
            "error_type": type(error).__name__,
            "message": str(error),
            "timestamp": datetime.now().isoformat()
        })

    def get_agent_dependencies(self, agent_name: str) -> Dict[str, Any]:
        """各エージェントが必要とする依存データを取得"""
        dependencies = {}
        
        if agent_name == "interactive":
            dependencies = {
                "persona": self.persona_data,
                "policy": self.policy_data
            }
        elif agent_name == "modeling":
            dependencies = {
                "video_path": self.video_path
            }
        elif agent_name == "goal_setting":
            dependencies = {
                "persona": self.persona_data,
                "policy": self.policy_data,
                "conversation_insights": self._extract_insights(),
                "motion_analysis": self.motion_analysis
            }
        elif agent_name == "plan":
            dependencies = {
                "goal": self.goals,
                "issues": self._extract_issues()
            }
        elif agent_name == "search":
            dependencies = {
                "tasks": self.training_plan.get("tasks", []),
                "player_level": self.persona_data.get("level", "初級")
            }
        elif agent_name == "summary":
            dependencies = {
                "analysis": self.motion_analysis,
                "goal": self.goals,
                "plan": self.training_plan
            }
            
        return dependencies

    def validate_dependencies(self, agent_name: str) -> bool:
        """エージェントの依存関係が満たされているかチェック"""
        dependencies = self.get_agent_dependencies(agent_name)
        return all(v is not None for v in dependencies.values())

    def _extract_insights(self) -> List[str]:
        """会話履歴から洞察を抽出"""
        insights = []
        for message in self.conversation_history:
            if message.get("role") == "assistant" and "insight" in message:
                insights.append(message["insight"])
        return insights

    def _extract_issues(self) -> List[str]:
        """動作分析から技術的課題を抽出"""
        if self.motion_analysis and "issues" in self.motion_analysis:
            return self.motion_analysis["issues"]
        return []