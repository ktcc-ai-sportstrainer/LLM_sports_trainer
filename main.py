from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from core.agent_state import AgentState

# agents, chains, core, modelsディレクトリ以下をまとめてimport
from chain.role_based_cooperation import RoleBasedCooperation
# utilsディレクトリ以下をimport
from utils.memory import ConversationBufferWindowMemory


# LLMの設定
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

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
    k=20,
    input_key="user_input",
    chat_history_key="chat_history",
    return_memory_prefix="",
)
output = StrOutputParser()

# プロンプトの設定
prompt = ChatPromptTemplate(
   [
       ("system", """あなたは野球のコーチです。ユーザーの入力と会話履歴からユーザーの悩みを深掘りするような質問をしてください。"""),
       MessagesPlaceholder("chat_history", optional=True),
       ("human", "{user_input}"),
   ]
)

# LLMChainの初期化
llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

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
        ("system", """あなたは文章を要約する達人です。会話履歴を以下のようにまとめてください。
         -----現在の悩み-----

        -----プロフィール-----

         """),
        ("human", "{history}"),
    ]
)

chain = summarization_prompt | llm | output
summary = chain.invoke({"history": memory.buffer})

# RoleBasedCooperationのインスタンス化
agent = RoleBasedCooperation(llm=llm)

# graphの生成 (デバッグ用に残す場合はコメントアウトを外す)
graph = agent._create_graph()
for s in graph.stream(AgentState(query=summary)):
    if "__end__" not in s:
        print(s)
        print("----")

result= agent.run(query=summary)
print(result)
