import operator
import datetime

from typing import Annotated,Any
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

llm = ChatOpenAI(
    model="gpt-4o", 
    temperature="0.7",

)


class Goal(BaseModel):
    description: str = Field(..., description="目標の設定")

    @property
    def text(self) -> str:
        return f"{self.description}"

class OptimizedGoal(BaseModel):
    description: str = Field(..., description="目標の説明")
    metrics: str = Field(..., description="目標の達成度を測定する方法")

    @property
    def text(self) -> str:
        return f"{self.description}(測定基準:{self.metrics})"
       
class Role(BaseModel):
    name: str = Field(..., description="役割の名前")
    description: str = Field(..., description="役割の詳細な説明")
    key_skills: list[str] = Field(..., description="この役割に必要な主要スキルや属性")

class Task(BaseModel):
    description: str = Field(..., description="タスクの説明")
    role: Role = Field(default=None, description="タスクに割り当てられた役割")

class TaskWithRoles(BaseModel):
    tasks: list[Task] = Field(...,description="役割が割り当てられたタスクのリスト")


class AgentState(BaseModel):
    query: str = Field(..., description="ユーザーが入力したクエリ")
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(default="", description="最適化されたレスポンス定義")
    tasks: list[Task] = Field(
        default_factory=list, description="実行するタスクのリスト"
    )
    current_task_index: int = Field(default=0, description="現在実行中のタスクの番号")
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="実行済みタスクの結果リスト"
    )
    final_report: str = Field(default="", description="最終的な出力結果")


class PassiveGoalCreator:
    def __init__(
            self,
            llm: ChatOpenAI
    ):
        self.llm = llm
    
    def run(self, query: str) -> Goal:
        prompt = ChatPromptTemplate.from_template(
            "ユーザーの入力を分析し、明確で実行可能な目標を生成してください。\n"
            "要件\n"
            "1. 目標は具体的かつ明確であり、実行可能なレベルで詳細化されている可能性があります。\n"
            "2. あなたが実行可能な行動は以下の通りです。\n"
            "  -インターネットを利用して、目標を達成するための調査を行う。\n"
            "  -ユーザーのためのレポートを生成する。\n"
            "3. 決して2.以外の行動を取ってはいけません。\n"
            "ユーザーの入力:{query}"
        )
        chain = prompt | self.llm.with_structured_output(Goal)
        return chain.invoke({"query": query})
    
class PromptOptimizer:
    def __init__(
            self,
            llm: ChatOpenAI
    ):
        self.llm = llm
    
    def run(self, query: str) -> OptimizedGoal:
        prompt = ChatPromptTemplate.from_template(
            """あなたは目標設定の専門家です。以下のユーザーの悩みとプロフィールからSMART原則(Specific: 具体的、Measurable: 測定可能、
            Achievable: 達成目標、Relavant: 関連性が高い、 Time-bound: 期限がある、)に基づいて最適化してください。\n\n"""
            "ユーザーの悩みとプロフィール:\n"
            "{query}\n\n"
            "指示:\n"
            "1. 元の目標を分析し、不足している要素や改善点を特定してください。\n"
            "2. あなたが実行可能なのは以下の行動だけです。\n"
            "  -インターネットを利用して、目標を達成するための調査を行う。\n"
            "  -ユーザーのためのレポートを生成する。\n"
            "3. SMART原則の各要素を考慮しながら、目標を具体的かつ詳細に記載してください。\n"
            "  -一切抽象的な表現を含んではいけません。\n"
            "  -必ずすべての単語が実行か可能かつ具体的であることを確認してください。\n"
            "4. 目標の達成度を測定する方法を具体的かつ詳細に記載してください。\n"
            "5. 元の目標で期限が指定されていない場合は、期限を考慮する必要はありません。\n"
            "6. REMEMBER: 決して2.以外の行動を取ってはいけません。"
        )
        chain = prompt | self.llm.with_structured_output(OptimizedGoal)
        return chain.invoke({"query": query})

class ResponseOptimizer:
    def __init__(
            self,
            llm: ChatOpenAI
    ):
        self.llm = llm
    
    def run(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """あなたはAIエージェントシステムのレスポンス最適化スペシャリストです。与えられた目標に対して、エージェントが
                    目標にあったレスポンスを返すためのレスポンス仕様を策定してください。"""
                ),
                (
                    "human",
                    "以下の手順に従って、レスポンス最適化プロンプトを作成してください:\n\n"
                    "1. 目標分析:\n"
                    "提示された目標を分析し、主要な要素や意図を特定してください。\n\n"
                    "2. レスポンス仕様の策定:\n"
                    """目標達成のための適切なレスポンス仕様を考案してください。トーン、構造、内容の焦点などを考慮に
                    入れてください。\n\n"""
                    "3. 具体的な指示の作成:\n"
                    """事前に収集された情報から、ユーザーの期待に沿ったレスポンスをするために必要な、AIエージェントに対する
                    明確で実行可能な指示を作成してください。あなたの指示によってAIエージェントが実行可能なのは、既に調査済み
                    の結果をまとめることだけです。インターネットへのアクセスはできません。\n\n"""
                    "4. 例の提供:\n"
                    "可能であれば、目標に沿ったレスポンスの例を1つ以上含めてください。\n\n"
                    "5. 評価基準の設定:\n"
                    "レスポンスの効果を測定するための基準を定義してください。\n\n"
                    "以下の構造でレスポンス最適化プロンプトを出力してください。:\n\n"
                    "目標分析:\n"
                    "[ここに目標の分析結果を記入]\n\n"
                    "レスポンス仕様:\n"
                    "[ここに策定されたレスポンスの仕様を記入]\n\n"
                    "AIエージェントへの指示:\n"
                    "[ここにAIエージェントへの具体的な指示を記入]\n\n"
                    "レスポンス例:\n"
                    "[ここにレスポンス例を記入]\n\n"
                    "評価基準:\n"
                    "[ここに評価基準を記入]\n\n"
                    "では、以下の目標に対するレスポンス最適化プロンプトを作成してください:\n"
                    "{query}",
                ),
           ]
       )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})


class DecomposedTasks(BaseModel):
    values: list[str] = Field(
        default_factory=list,
        min_items=3,
        max_items=5,
        description="3～5個に分解されたタスク"
    )

class QueryDecomposer:
    def __init__(self, llm:ChatOpenAI):
        self.llm = llm
        self.current_data = datetime.datetime.now().strftime("%Y-%m-%d")
    def run(self, query: str) -> DecomposedTasks:
        prompt = ChatPromptTemplate.from_template(
            f"CURRENT_DATE: {self.current_data}\n"
            "-----\n"
            """タスク: 与えられた目標を具体的で実行可能なタスクに分解してください。特に重要なタスクは練習メニューを作る
            ことと、ユーザーの悩みに応じた人間味のある応援を考えることです。\n"""
            "要件:\n"
            "1. 以下の行動だけで目標を達成すること。決して指定された行動をとらないこと。\n"
            "   - インターネットを利用して、目標を達成するための調査を行う。\n"
            "2. 各タスクは具体的かつ詳細に記載されており、単独で実行並びに検証可能な情報を含めること。一切抽象的な表現を含まないこと。\n"
            "3. タスクは実行可能な順序でリスト化すること。\n"
            "4. 特に重要なタスクは最優先で必ず実行してください。\n"
            "5. タスクは日本語で出力すること。\n"
            "目標: {query}"
        )
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.invoke({"query": query}) # これを追加

class planner:
    def __init__(self, llm:ChatOpenAI):
        self.query_decomposer = QueryDecomposer(llm=llm)

    def run(self, query:str) -> list[Task]:
        decomposed_tasks: DecomposedTasks = self.query_decomposer.run(query=query)
        return [Task(description=task) for task in decomposed_tasks.values]

class RoleAssigner:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(TaskWithRoles)
    
    def run(self, tasks: list[Task]) -> list[Task]:
        prompt = ChatPromptTemplate(
            [
                (
                    "system",
                    (
                        "あなたは創造的な役割設計の専門家です。与えられたタスクに対して、ユニークで適切な役割を生成してください。"
                    ),
                ),
                (
                    "human",
                    (
                        "タスク:\n{tasks}\n\n"
                        "これらのタスクに対して、以下の指示に従って役割を割り当ててください:\n"
                        "1. 各タスクに対して、独自の創造的な役割を考案してください。既存の職業名や一般的な役割名にとらわれる必要はありません。\n"
                        "2. 役割名は、そのタスクの本質を反映した魅力的で記憶に残るものにしてください。\n"
                        "3. 各役割に対して、その役割がなぜそのタスクに最適なのかを詳細に説明してください。\n"
                        "4. その役割が効果的にタスクを遂行するために必要な主要なスキルやアトリビュートを3つ挙げてください。 \n\n"
                        "創造性を発揮し、タスクの本質を捉えた革新的な役割を生成してください。"
                    ),
                ),
            ],
        )
        chain = prompt | self.llm
        tasks_with_roles = chain.invoke(
            {"tasks": "\n".join([task.description for task in tasks])}
        )
        return tasks_with_roles.tasks
    
class Executor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.tools = [TavilySearchResults(max_results=3)]
        self.base_agent = create_react_agent(self.llm, self.tools)
    
    def run(self, task: Task) -> str:
        result = self.base_agent.invoke(
            {
                "messages":[
                    (
                        "system",
                        (
                            f"あなたは{task.role.name}です。\n"
                            f"説明:{task.role.description}\n"
                            f"主要なスキル:{','.join(task.role.key_skills)}\n"
                            "あなたの役割に基づいて、与えられたタスクを最高の能力で遂行してくください。"
                        ),
                    ),
                    (
                        "human",
                        f"以下のタスクを実行してください:\n\n{task.description}",
                    ),
                ]
            }
        )
        return result["messages"][-1].content

class Reportor: 
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    def run(self, query: str, response_definition: str, results: list[str]) -> str:
        prompt = ChatPromptTemplate.from_template(
            "与えられた目標:\n{query}\n\n"
            "調査結果:\n{results}\n\n"
            "与えられた目標に対し、調査結果を用いて、以下の指示に基づいてレスポンスを生成してください。"

            "{response_definition}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "query": query,
                "results": "\n\n".join(
                    f"Info{i+1}:\n{result}" for i, result in enumerate(results)
                ),
                "response_definition": response_definition, # これを追加
            }
        )

class RoleBasedCooperation:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.passive_goal_creator = PassiveGoalCreator(llm=llm)
        self.prompt_optimizer = PromptOptimizer(llm=llm)
        self.response_optimizer = ResponseOptimizer(llm=llm)
        self.planner = planner(llm=llm)
        self.role_assigner = RoleAssigner(llm=llm)
        self.executor = Executor(llm=llm)
        self.reporter = Reportor(llm=llm)
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        workflow.add_node("goal_setting", self._goal_setting)
        workflow.add_node("planner", self._plan_tasks)
        workflow.add_node("role_assigner", self._assign_roles)
        workflow.add_node("executor", self._execute_task)
        workflow.add_node("reporter",self._generate_report)

        workflow.set_entry_point("goal_setting")

        workflow.add_edge("goal_setting", "planner")
        workflow.add_edge("planner", "role_assigner")
        workflow.add_edge("role_assigner","executor")
        workflow.add_conditional_edges(
            "executor",
            lambda state: state.current_task_index < len(state.tasks),
            {True: "executor", False: "reporter"},
            )
    
        workflow.add_edge("reporter", END)

        return workflow.compile()
    
    def _goal_setting(self, state:AgentState) -> dict[str, Any]:
        goal: Goal = self.passive_goal_creator.run(query=state.query)
        optimized_goal: OptimizedGoal = self.prompt_optimizer.run(query=goal.text)
        optimized_response: str = self.response_optimizer.run(query=optimized_goal.text)

        return{
            "optimized_goal": optimized_goal.text,
            "optimized_response": optimized_response,
        }
    
    
    def _plan_tasks(self, state: AgentState) -> dict[str, Any]:
        tasks = self.planner.run(query=state.optimized_goal)
        return {"tasks": tasks}
   
    
    def _assign_roles(self, state: AgentState) -> dict[str, Any]:
        tasks_with_roles = self.role_assigner.run(tasks=state.tasks)
        return {"tasks": tasks_with_roles}
    

    def _execute_task(self, state:AgentState) -> dict[str, Any]:
        current_task = state.tasks[state.current_task_index]
        result = self.executor.run(task=current_task)
        return {
            "results": [result],
            "current_task_index":state.current_task_index + 1,
        }
    
  
    def _generate_report(self, state:AgentState) -> dict[str, Any]:
        report = self.reporter.run(
            query=state.optimized_goal,
            response_definition=state.optimized_response,
            results=state.results
        )
        return{"final_report": report}
    
    
    def run(self, query: str) -> str:
        initial_state = AgentState(query=query)

        final_state = self.graph.invoke(initial_state,{"recursion_limit": 1000})
        return final_state["final_report"]


# 初期質問リスト
initial_questions = [
    "現在のスイングで、何か気になる点や悩みはありますか？",
    "現在のバッティングレベルはどのくらいですか？（初心者、経験者、上級者など)",
    "性別を教えてください",
    "現在何歳ですか？",
    "身長と体重を教えてください。",
    "どのような打席結果が多いですか？（ヒット、凡打の種類、三振など）",
    "どんなタイプの打者を目指していますか？（長距離、アベレージ、チャンスメーカーなど）",
    "打球はどの方向に飛ぶ傾向がありますか",
    "どの球種が得意/苦手ですか？",
    "練習頻度はどのくらいですか？(週何回、何時から何時までなど)",
    "現在の練習メニューをなるべく詳細に教えてください。",
]

# メモリの初期化
memory = ConversationBufferWindowMemory(
    llm=llm,
    k=20,  # 過去のやり取りをいくつ保持するか
    input_key="user_input", 
    chat_history_key="chat_history",
    return_memory_prefix = "" #追記
)
output = StrOutputParser()


prompt = ChatPromptTemplate(
   [
       ("system","""あなたは野球のコーチです。ユーザーの入力と会話履歴からユーザーの悩みを深掘りするような質問をして
        ください。"""),
       MessagesPlaceholder("chat_history", optional=True),
       ("human","{user_input}"),
   ]
)

# LLMChainの初期化
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory,)

# 質問と回答のループ
for i, initial_question in enumerate(initial_questions):
    if i == 0:
        print(f"AI: {initial_question}")
        user_input = input("You: ")
        # memory.save_context() を使用してメモリに保存
        memory.save_context({"user_input": initial_question}, {"output": user_input})
        # 深堀り質問
        deep_question =llm_chain.invoke({"user_input":user_input,"chat_history":memory.buffer})
        print(f"AI: {deep_question['text'].strip()}")
        user_input2 = input("You: ")
      
        deep_question2 =llm_chain.invoke({"user_input":user_input2,"chat_history":memory.buffer})
        print(f"AI: {deep_question2['text'].strip()}")
        user_input3 = input("You: ")
        memory.save_context({"user_input": user_input3}, {"output": "fin"})
    else:
        print(f"AI: {initial_question}")
        user_input4= input("You: ")
        memory.save_context({"user_input": initial_question}, {"output": user_input4})

# 会話履歴の要約を生成
summarization_prompt = ChatPromptTemplate(
    [
        ("""system","あなたは文章を要約する達人です。会話履歴を以下のようにまとめてください。
         -----現在の悩み-----
         
        -----プロフィール-----
         
         """),
        ("human","{history}"),
    ]
    )

chain = summarization_prompt | llm | output
summary = chain.invoke({"history": memory.buffer}) 

print(summary)
agent = RoleBasedCooperation(llm=llm)

# graphの生成 (デバッグ用に残す場合はコメントアウトを外す)
graph = agent._create_graph()
for s in graph.stream(AgentState(query=summary)):
    if "__end__" not in s:
        print(s)
        print("----")

result= agent.run(query=summary)
print(result)













