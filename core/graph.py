from typing import Dict, Any, List, Callable, Awaitable
from langchain.graphs import Graph
import networkx as nx
from datetime import datetime

from core.state import SystemState
from core.logger import SystemLogger

class WorkflowGraph:
    """エージェント間の依存関係とワークフローを管理するクラス"""
    
    def __init__(self, state: SystemState, logger: SystemLogger):
        self.state = state
        self.logger = logger
        self.graph = self._create_workflow_graph()
        
    def _create_workflow_graph(self) -> nx.DiGraph:
        """ワークフローグラフの作成"""
        G = nx.DiGraph()
        
        # ノードの追加（エージェント）
        agents = [
            "interactive",
            "modeling",
            "goal_setting",
            "plan",
            "search",
            "summary"
        ]
        
        for agent in agents:
            G.add_node(agent)
        
        # エッジの追加（依存関係）
        edges = [
            ("interactive", "goal_setting"),
            ("modeling", "goal_setting"),
            ("goal_setting", "plan"),
            ("plan", "search"),
            ("modeling", "summary"),
            ("goal_setting", "summary"),
            ("plan", "summary")
        ]
        
        for edge in edges:
            G.add_edge(*edge)
        
        return G

    async def execute_workflow(
        self,
        agent_executors: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
    ) -> None:
        """ワークフローの実行"""
        # 実行順序の決定
        execution_order = list(nx.topological_sort(self.graph))
        
        for agent_name in execution_order:
            self.logger.log_info(f"Starting {agent_name} agent execution")
            
            try:
                # 依存関係のチェック
                if not self.state.validate_dependencies(agent_name):
                    raise ValueError(f"Dependencies not met for {agent_name}")
                
                # エージェントの実行
                dependencies = self.state.get_agent_dependencies(agent_name)
                result = await agent_executors[agent_name](dependencies)
                
                # 結果の保存
                self.state.add_agent_result(agent_name, result)
                self.state.update_agent_status(agent_name, True)
                
                self.logger.log_info(f"Completed {agent_name} agent execution")
                
            except Exception as e:
                self.state.add_error(agent_name, e)
                self.logger.log_error(f"Error in {agent_name} agent: {str(e)}")
                raise

    def get_agent_dependencies(self, agent_name: str) -> List[str]:
        """特定のエージェントの依存関係を取得"""
        predecessors = list(self.graph.predecessors(agent_name))
        return predecessors

    def get_dependent_agents(self, agent_name: str) -> List[str]:
        """特定のエージェントに依存している他のエージェントを取得"""
        successors = list(self.graph.successors(agent_name))
        return successors

    def check_cycle(self) -> bool:
        """依存関係の循環をチェック"""
        return nx.is_directed_acyclic_graph(self.graph)

    def validate_workflow(self) -> bool:
        """ワークフローの妥当性を検証"""
        # 循環依存のチェック
        if not self.check_cycle():
            self.logger.log_error("Circular dependency detected in workflow")
            return False
            
        # 孤立ノードのチェック
        if len(list(nx.isolates(self.graph))) > 0:
            self.logger.log_error("Isolated agents detected in workflow")
            return False
            
        # 到達可能性のチェック
        if not nx.is_weakly_connected(self.graph):
            self.logger.log_error("Workflow graph is not fully connected")
            return False
            
        return True

    def get_execution_status(self) -> Dict[str, Any]:
        """ワークフロー実行状態の取得"""
        status = {
            "completed_agents": [],
            "pending_agents": [],
            "failed_agents": [],
            "current_timestamp": datetime.now().isoformat()
        }
        
        for agent in self.graph.nodes():
            agent_status = f"{agent.lower()}_completed"
            if hasattr(self.state, agent_status):
                if getattr(self.state, agent_status):
                    status["completed_agents"].append(agent)
                else:
                    if any(e["agent"] == agent for e in self.state.errors):
                        status["failed_agents"].append(agent)
                    else:
                        status["pending_agents"].append(agent)
                        
        return status