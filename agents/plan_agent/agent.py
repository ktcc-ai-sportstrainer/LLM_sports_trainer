from typing import Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from agents.base import BaseAgent
from models.internal.goal import Goal
from models.internal.plan import TrainingPlan, TrainingTask

class PlanAgent(BaseAgent):
    """
    GoalSettingAgentによって設定された目標を受け取り、
    練習タスクやステップを提案するエージェント。
    """

    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        self.prompt = ChatPromptTemplate.from_template(
            """あなたは熟練の野球コーチです。以下の目標と課題を達成するための練習計画を提案してください。

            【目標】
            主目標: {primary_goal}
            サブ目標:
            {sub_goals}

            【技術的課題】
            {issues}

            出力フォーマット:
            - タスクのリスト（タイトル、詳細内容、所要時間、注意点、必要な道具）
            - 全体のステップアップの道筋
            - 全体での必要時間

            なるべく具体的に、かつ実践的に提案してください。
            """
        )

    async def run(
        self,
        goal: Goal,
        issues: list[str]
    ) -> Dict[str, Any]:
        """
        目標と技術的課題を基に、練習計画を提案する。
        """
        sub_goals_str = "\n".join(goal.sub_goals)
        issues_str = "\n".join(issues)

        response = await self.llm.ainvoke(
            self.prompt.format(
                primary_goal=goal.primary_goal,
                sub_goals=sub_goals_str,
                issues=issues_str
            )
        )

        # テキストからPlanをパース（簡易パース）
        training_plan = self._parse_plan(response.content)

        return self.create_output(
            output_type="training_plan",
            content=training_plan.dict()
        ).dict()

    def _parse_plan(self, text: str) -> TrainingPlan:
        """
        LLM応答を読み取り、TrainingPlanモデルに変換する簡易実装例。
        """
        # 非常に簡易的に行頭をみてsplitしたり、正規表現を使ったりできる。
        # ここではサンプルとして固定的に分割しているが、実際にはプロンプト設計やjson出力を使うのが望ましい。

        # ダミー例
        tasks = [
            TrainingTask(
                title="素振りドリル",
                description="正しいフォームを意識した素振りを10〜15分行う。鏡を使って自分のフォームをチェックする。",
                duration="15分",
                focus_points=["フォーム維持", "グリップの再確認"],
                equipment=["バット", "鏡"]
            ),
            TrainingTask(
                title="ティーバッティング",
                description="ティーに置かれたボールを的確に捉える練習。インコース・アウトコースに分けて実施。",
                duration="20分",
                focus_points=["インサイドアウト", "下半身の使い方"],
                equipment=["バット", "ティースタンド", "ボール"]
            )
        ]

        plan = TrainingPlan(
            tasks=tasks,
            progression_path="最初はフォーム矯正、次にタイミング合わせを重点に、最後に実践形式へ移行",
            required_time="1時間程度"
        )
        return plan
