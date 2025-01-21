from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate

class SearchAgent(BaseAgent):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        self.prompt = ChatPromptTemplate.from_template(
            """以下の練習タスクに関連する具体的な練習メニューと指導のポイントを提案してください。

            タスク: {task}
            目的: {purpose}
            選手レベル: {level}
            
            以下の要素を含めて提案してください：
            1. 具体的な練習メニュー（時間配分含む）
            2. 段階的な難易度設定
            3. 重要な指導ポイント
            4. 必要な道具
            5. 注意事項
            """
        )

    async def run(
        self,
        tasks: List[Dict[str, Any]],
        player_level: str
    ) -> Dict[str, Any]:
        """
        各タスクに対して具体的な練習メニューを生成
        """
        detailed_tasks = []
        
        for task in tasks:
            # タスクごとに詳細な練習メニューを生成
            response = await self.llm.ainvoke(
                self.prompt.format(
                    task=task["title"],
                    purpose=task["description"],
                    level=player_level
                )
            )
            
            # レスポンスをパース
            menu = await self._parse_training_menu(response.content)
            
            # 元のタスク情報と結合
            detailed_task = {
                **task,
                "training_menu": menu
            }
            detailed_tasks.append(detailed_task)
        
        return self.create_output(
            output_type="detailed_training_tasks",
            content={"tasks": detailed_tasks}
        ).dict()

    async def _parse_training_menu(self, content: str) -> Dict[str, Any]:
        """
        生成されたテキストから構造化された練習メニューを抽出
        """
        parse_prompt = ChatPromptTemplate.from_template(
            """以下の練習メニュー提案を構造化データに変換してください：
            
            {content}
            
            以下の形式でJSON形式の出力を生成してください：
            {
                "drills": [
                    {
                        "name": "練習名",
                        "duration": "所要時間",
                        "description": "詳細説明",
                        "key_points": ["指導ポイント1", "指導ポイント2"],
                        "equipment": ["必要な道具1", "必要な道具2"],
                        "cautions": ["注意点1", "注意点2"]
                    }
                ],
                "progression_steps": ["ステップ1", "ステップ2"],
                "total_time": "合計時間"
            }
            """
        )
        
        response = await self.llm.ainvoke(
            parse_prompt.format(content=content)
        )
        
        # 文字列をJSONとしてパース
        try:
            menu = json.loads(response.content)
        except json.JSONDecodeError:
            # パースに失敗した場合は簡易的な構造を返す
            menu = {
                "drills": [],
                "progression_steps": [],
                "total_time": "N/A"
            }
        
        return menu