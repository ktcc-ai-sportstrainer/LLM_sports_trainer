from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from langgraph.graph import StateGraph

from core.state import AgentState, StateManager
from core.logger import SystemLogger
from agents.base import BaseAgent

class Executor:
    """エージェントの実行を管理するクラス"""

    def __init__(self, 
                 agents: Dict[str, BaseAgent],
                 logger: SystemLogger):
        self.agents = agents
        self.logger = logger
        self.state_manager = None
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """LangGraphベースのワークフローを構築"""
        graph = StateGraph(AgentState)
        
        # エージェントノードの追加
        for name, agent in self.agents.items():
            graph.add_node(name, self._create_agent_handler(name, agent))
        
        # エッジの定義（実行順序の設定）
        graph.add_edge("interactive", "modeling")
        graph.add_edge("modeling", "goal_setting")
        graph.add_edge("goal_setting", "plan")
        graph.add_edge("plan", "search")
        graph.add_edge("search", "summary")
        
        # 開始・終了ポイントの設定
        graph.set_entry_point("interactive")
        graph.set_finish_point("summary")
        
        return graph.compile()

    def _create_agent_handler(self, name: str, agent: BaseAgent):
        """エージェント実行用のハンドラを生成"""
        async def handler(state: AgentState) -> AgentState:
            try:
                self.logger.log_info(f"Starting {name} agent", agent=name)
                
                # エージェントの入力を準備
                input_data = self.state_manager.get_agent_input(name)
                
                # エージェントの実行
                start_time = datetime.now()
                result = await agent.run(**input_data)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # 実行メトリクスの記録
                self.logger.log_info(
                    f"Agent {name} completed in {execution_time:.2f} seconds",
                    agent=name
                )
                
                # 状態の更新
                return self.state_manager.update_state(name, result)
                
            except Exception as e:
                self.logger.log_error_details(error=e, agent=name)
                return self.state_manager.update_state(name, {}, error=e)
                
        return handler

    async def execute(self,
                     initial_state: AgentState,
                     timeout: Optional[float] = None) -> Dict[str, Any]:
        """ワークフローを実行"""
        try:
            # StateManagerの初期化
            self.state_manager = StateManager(initial_state)
            
            # タイムアウト付きで実行
            if timeout:
                try:
                    final_state = await asyncio.wait_for(
                        self.workflow.ainvoke(initial_state),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    raise ExecutionError("Workflow execution timed out")
            else:
                final_state = await self.workflow.ainvoke(initial_state)
            
            return self._format_result(final_state)
            
        except Exception as e:
            self.logger.log_error_details(
                error=e,
                context={"initial_state": initial_state}
            )
            raise ExecutionError(f"Workflow execution failed: {str(e)}")

    def _format_result(self, state: AgentState) -> Dict[str, Any]:
        """実行結果を整形"""
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
                "execution_metrics": self.state_manager.get_execution_status()
            }
        }

    def get_execution_status(self) -> Dict[str, Any]:
        """実行状態を取得"""
        if self.state_manager:
            return self.state_manager.get_execution_status()
        return {
            "status": "not_started",
            "error_count": 0,
            "completed_steps": []
        }

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """実行履歴を取得"""
        if self.state_manager:
            return self.state_manager.history
        return []