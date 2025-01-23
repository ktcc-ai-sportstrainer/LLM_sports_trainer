from datetime import datetime
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from agents import (
    InteractiveAgent,
    ModelingAgent,
    GoalSettingAgent,
    PlanAgent,
    SearchAgent,
    SummarizeAgent
)
from core.executor import Executor
from core.state import AgentState
from core.logger import SystemLogger

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

class SwingCoachingSystem:
    """野球スイングコーチングシステム全体を制御するクラス"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = SystemLogger()
        
        # LLMの初期化
        self.llm = ChatOpenAI(
            openai_api_key=config.get("openai_api_key"),
            model=config.get("model_name", "gpt-4"),
            temperature=0.7
        )
        
        # エージェントの初期化
        self.agents = {
            "interactive": InteractiveAgent(self.llm),
            "modeling": ModelingAgent(self.llm),
            "goal_setting": GoalSettingAgent(self.llm),
            "plan": PlanAgent(self.llm),
            "search": SearchAgent(self.llm),
            "summary": SummarizeAgent(self.llm)
        }
        
        # ワークフローの初期化
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """LangGraphベースのワークフローを構築"""
        graph = StateGraph(AgentState)
        
        # エージェントノードの追加
        for name, agent in self.agents.items():
            graph.add_node(name, self._create_agent_handler(name, agent))
        
        # エッジの定義
        graph.add_edge("interactive", "modeling")
        graph.add_edge("modeling", "goal_setting")
        graph.add_edge("goal_setting", "plan")
        graph.add_edge("plan", "search")
        graph.add_edge("search", "summary")
        
        # 開始・終了ポイントの設定
        graph.set_entry_point("interactive")
        graph.set_finish_point("summary")
        
        return graph.compile()

    def _create_agent_handler(self, name: str, agent: Any):
        """エージェント実行用のハンドラを生成"""
        async def handler(state: AgentState) -> AgentState:
            try:
                self.logger.log_info(f"Starting {name} agent", agent=name)
                
                # エージェントの実行
                result = await agent.run(state)
                
                # 状態の更新
                new_state = {
                    **state,
                    "last_agent": name,
                    "status": "completed"
                }
                
                # エージェント固有の出力を追加
                if name == "interactive":
                    new_state["conversation"] = result.get("conversation", [])
                elif name == "modeling":
                    new_state["motion_analysis"] = result.get("motion_analysis", {})
                elif name == "goal_setting":
                    new_state["goals"] = result.get("goals", {})
                elif name == "plan":
                    new_state["plan"] = result.get("plan", {})
                elif name == "search":
                    new_state["resources"] = result.get("resources", {})
                elif name == "summary":
                    new_state["summary"] = result.get("summary", "")
                
                self.logger.log_info(f"Completed {name} agent", agent=name)
                return new_state
                
            except Exception as e:
                self.logger.log_error_details(error=e, agent=name)
                return {
                    **state,
                    "status": "error",
                    "errors": state.get("errors", []) + [{
                        "agent": name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }]
                }
        
        return handler

    async def run(
        self,
        persona_data: Dict[str, Any],
        policy_data: Dict[str, Any],
        user_video_path: str,
        ideal_video_path: str
    ) -> Dict[str, Any]:
        """システムを実行し、最終結果を返す"""
        try:
            # 初期状態の設定
            initial_state: AgentState = {
                "persona_data": persona_data,
                "policy_data": policy_data,
                "user_video_path": user_video_path,
                "ideal_video_path": ideal_video_path,
                "conversation": [],
                "motion_analysis": {},
                "goals": {},
                "plan": {},
                "resources": {},
                "summary": "",
                "status": "started",
                "errors": [],
                "last_agent": ""
            }
            
            # ワークフローの実行
            self.logger.log_info("Starting workflow execution")
            start_time = datetime.now()
            
            final_state = await self.workflow.ainvoke(initial_state)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.log_info(f"Workflow completed in {execution_time:.2f} seconds")
            
            # 結果の整形
            return self._format_result(final_state)
            
        except Exception as e:
            self.logger.log_error_details(
                error=e,
                context={
                    "persona_data": persona_data,
                    "policy_data": policy_data,
                    "user_video_path": user_video_path,
                    "ideal_video_path": ideal_video_path
                }
            )
            raise

    def _format_result(self, state: AgentState) -> Dict[str, Any]:
        """最終結果を整形"""
        return {
            "interactive_questions": [
                msg["question"] for msg in state["conversation"]
                if msg.get("role") == "assistant" and "question" in msg
            ],
            "motion_analysis": state["motion_analysis"],
            "goal_setting": state["goals"],
            "training_plan": state["plan"],
            "search_results": state["resources"],
            "final_summary": {
                "summary": state["summary"],
                "execution_metrics": {
                    "status": state["status"],
                    "error_count": len(state["errors"]),
                    "last_agent": state["last_agent"]
                }
            }
        }