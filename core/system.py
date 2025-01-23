from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict
import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from agents import (
    InteractiveAgent,
    ModelingAgent,
    GoalSettingAgent,
    PlanAgent,
    SearchAgent,
    SummarizeAgent
)
from core.state import AgentState, create_initial_state
from core.logger import SystemLogger
from models.input.persona import Persona
from models.input.policy import TeachingPolicy

class SwingCoachingSystem:
    """野球スイングコーチングシステム全体を制御するクラス"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = SystemLogger()

        # OpenAI APIキーの取得と検証
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # LLMの初期化
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
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

        # ワークフローの構築 (コンストラクタ内で直接定義)
        self.workflow = StateGraph(AgentState)
        for name, agent in self.agents.items():
            node_name = f"{name}_node"
            self.workflow.add_node(node_name, self._create_agent_handler(name, agent))
        self.workflow.add_edge("interactive_node", "modeling_node")
        self.workflow.add_edge("modeling_node", "goal_setting_node")
        self.workflow.add_edge("goal_setting_node", "plan_node")
        self.workflow.add_edge("plan_node", "search_node")
        self.workflow.add_edge("search_node", "summary_node")
        self.workflow.set_entry_point("interactive_node")
        self.workflow.set_finish_point("summary_node")
        self.workflow = self.workflow.compile()

    def _create_agent_handler(self, name: str, agent: Any):
        """エージェント実行用のハンドラを生成"""
        async def handler(state: AgentState) -> AgentState:
            try:
                self.logger.log_info(f"Starting {name} agent", agent=name)

                # エージェントの入力を準備
                input_data = self._prepare_agent_input(name, state)

                # エージェントの実行
                start_time = datetime.now()
                result = await agent.run(**input_data)
                execution_time = (datetime.now() - start_time).total_seconds()
                self.logger.log_info(f"Agent {name} completed in {execution_time:.2f} seconds", agent=name)

                # 状態の更新
                new_state = {
                    **state,
                    "last_agent": name,
                    "status": "completed"
                }
                if result:
                    new_state.update(result)

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

    def _prepare_agent_input(self, agent_name: str, state: AgentState) -> Dict[str, Any]:
        """エージェント実行のために必要な入力データを準備"""
        if agent_name == "interactive":
            return {
                "persona": Persona(**state["persona_data"]),
                "policy": TeachingPolicy(**state["policy_data"]),
                "conversation_history": state.get("conversation", [])
            }
        elif agent_name == "modeling":
            return {
                "user_video_path": state["user_video_path"],
                "ideal_video_path": state["ideal_video_path"]
            }
        elif agent_name == "goal_setting":
            return {
                "persona": Persona(**state["persona_data"]),
                "policy": TeachingPolicy(**state["policy_data"]),
                "conversation_insights": state.get("conversation", []),
                "motion_analysis": state.get("motion_analysis", {})
            }
        elif agent_name == "plan":
            return {
                "goal": state.get("goals", {}),
                "issues": state.get("motion_analysis", {}).get("user_analysis", {}).get("issues", [])
            }
        elif agent_name == "search":
            return {
                "tasks": state.get("plan", {}).get("tasks", []),
                "player_level": state["persona_data"]["level"]
            }
        elif agent_name == "summary":
            return {
                "analysis": state.get("motion_analysis", {}),
                "goal": state.get("goals", {}),
                "plan": state.get("plan", {})
            }
        else:
            raise ValueError(f"Unknown agent: {agent_name}")

    async def run(
        self,
        persona_data: Dict[str, Any],
        policy_data: Dict[str, Any],
        user_video_path: str,
        ideal_video_path: str
    ) -> Dict[str, Any]:
        """システムを実行し、最終結果を返す"""
        try:
            # ビデオファイルの存在確認
            for video_path in [user_video_path, ideal_video_path]:
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")

            # 初期状態の作成
            initial_state = create_initial_state(
                persona_data=persona_data,
                policy_data=policy_data,
                user_video_path=user_video_path,
                ideal_video_path=ideal_video_path
            )

            # 実行開始時刻の記録
            initial_state["start_time"] = datetime.now()

            # ワークフローの実行
            self.logger.log_info("Starting workflow execution")
            final_state = await self.workflow.ainvoke(initial_state)

            # 実行時間の計算と記録
            execution_time = (datetime.now() - initial_state["start_time"]).total_seconds()
            self.logger.log_info(f"Workflow completed in {execution_time:.2f} seconds")

            # 結果の整形と返却
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
            "interactive_questions": state.get("interactive_questions", []),
            "motion_analysis": state.get("motion_analysis", {}),
            "goal_setting": state.get("goals", {}),
            "training_plan": state.get("plan", {}),
            "search_results": state.get("resources", {}),
            "final_summary": {
                "summary": state.get("summary", ""),
                "execution_metrics": {
                    "status": state.get("status", "unknown"),
                    "error_count": len(state.get("errors", [])),
                    "last_agent": state.get("last_agent", ""),
                    "total_time": (
                        datetime.now() - state.get("start_time", datetime.now())
                    ).total_seconds() if "start_time" in state else 0
                }
            }
        }
