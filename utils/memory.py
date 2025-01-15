from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI

# このファイルでは、ConversationBufferWindowMemoryを初期化する関数を提供します。
# 必要に応じて、他のメモリ関連のユーティリティ関数もここに追加できます。

def create_memory(llm: ChatOpenAI, k: int = 20) -> ConversationBufferWindowMemory:
    """ConversationBufferWindowMemoryを初期化して返します。

    Args:
        llm: 使用するLLM。
        k: 過去のやり取りをいくつ保持するか。

    Returns:
        初期化されたConversationBufferWindowMemory。
    """
    memory = ConversationBufferWindowMemory(
        llm=llm,
        k=k,
        input_key="user_input",
        chat_history_key="chat_history",
        return_memory_prefix="",
    )
    return memory