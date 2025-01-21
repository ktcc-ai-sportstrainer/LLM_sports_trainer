# core/system.py

from typing import Dict, Any
import traceback
from datetime import datetime
import asyncio

from langchain_openai import ChatOpenAI
from core.state import SystemState
from core.graph import WorkflowGraph
from core.logger import SystemLogger

from agents.interactive_agent import InteractiveAgent
from agents.modeling_agent import ModelingAgent
from agents.goal_setting_agent import GoalSettingAgent
from agents.plan_agent import PlanAgent
from agents.search_agent import SearchAgent
from agents.summarize_agent import SummarizeAgent

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

        # 各エージェントのロガー設定
        for agent_name in self.agents:
            self.logger.setup_agent_logger(agent_name)

    async def run(
        self,
        persona_data: Dict[str, Any],
        policy_data: Dict[str, Any],
        user_video_path: str,
        ideal_video_path: str
    ) -> Dict[str, Any]:
        """
        2本の動画(ユーザーのスイング / 理想スイング)を含めて
        システムを実行し、最終結果を返す。
        """
        try:
            # SystemStateの初期化
            state = SystemState(
                persona_data=persona_data,
                policy_data=policy_data,
                user_video_path=user_video_path,
                ideal_video_path=ideal_video_path
            )

            # ワークフローグラフの初期化
            workflow = WorkflowGraph(state, self.logger)

            # ワークフローの妥当性検証
            if not workflow.validate_workflow():
                raise ValueError("Invalid workflow configuration (circular or disconnected)")

            # エージェント実行関数を準備
            agent_executors = {
                name: self._create_executor(agent)
                for name, agent in self.agents.items()
            }

            # ワークフローの実行
            start_time = datetime.now()
            self.logger.log_info("Starting workflow execution")

            await workflow.execute_workflow(agent_executors)

            # 実行時間
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.log_info(f"Workflow completed in {execution_time:.2f} seconds")

            # 最終結果をまとめ
            result = self._prepare_final_result(state)
            return result

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

    def _create_executor(self, agent: Any):
        """エージェント実行関数の生成"""
        async def execute(dependencies: Dict[str, Any]) -> Dict[str, Any]:
            agent_name = agent.__class__.__name__.lower().replace("agent", "")
            try:
                self.logger.log_info(f"Executing {agent_name}", agent=agent_name)
                self.logger.log_agent_input(agent_name, dependencies)

                start_time = datetime.now()
                result = await agent.run(**dependencies)
                execution_time = (datetime.now() - start_time).total_seconds()

                self.logger.log_agent_output(agent_name, result)
                self.logger.log_execution_time(agent_name, execution_time)
                return result

            except Exception as e:
                self.logger.log_error_details(error=e, agent=agent_name, context=dependencies)
                raise

        return execute

    def _prepare_final_result(self, state: SystemState) -> Dict[str, Any]:
        """最終結果の辞書を作成"""
        result = {
            "interactive_questions": self._extract_questions(state),
            "motion_analysis": state.motion_analysis,
            "goal_setting": state.goals,
            "training_plan": state.training_plan,
            "search_results": state.search_results,
            "final_summary": state.final_summary,
            "execution_metrics": {
                "completed_agents": [
                    agent for agent in self.agents
                    if getattr(state, f"{agent}_completed")
                ],
                "error_count": len(state.errors),
                "conversation_turns": len(state.conversation_history)
            }
        }
        return result

    def _extract_questions(self, state: SystemState) -> List[str]:
        """対話エージェントの質問を抽出(例)"""
        questions = []
        for msg in state.conversation_history:
            # ここでは仮に "assistant" かつ "question"キーがあるものを抽出
            if msg.get("role") == "assistant" and "question" in msg:
                questions.append(msg["question"])
        return questions

