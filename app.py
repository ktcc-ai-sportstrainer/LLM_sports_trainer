import streamlit as st
from config import load_config
from core.system import SwingCoachingSystem
import os
import json
import asyncio
from agents.interactive_agent.agent import InteractiveAgent

def get_streamlit_user_answer(question: str) -> str:
    """
    Streamlitでユーザーからの回答を受け取る。
    ここではシンプルに `st.text_input` を用いる例。
    実際にはUI上で「次へ」ボタンなどを押さないとループできないなど工夫が必要。
    """
    st.write(f"### 質問: {question}")
    user_input = st.text_input("あなたの回答", key=question)  # keyにquestionを使うのは簡易例
    # Streamlit はリアルタイム対話に少し工夫が必要。ここでは極簡易例として示す。
    return user_input if user_input else "（未入力）"

def main():
    st.title("野球スイングコーチングAI - Streamlit Demo")

    # 1. JSON入力
    st.sidebar.header("JSONファイルのアップロード")
    json_file = st.sidebar.file_uploader("ペルソナ情報 JSONファイル", type=["json"])

    # 2. 動画ファイル
    st.write("## あなたのスイング動画")
    user_uploaded_file = st.file_uploader("アップロードしてください", type=["mp4","mov","avi"])
    st.write("## 理想のスイング動画（任意）")
    ideal_uploaded_file = st.file_uploader("理想のスイング動画をアップロード", type=["mp4","mov","avi"])

    if st.button("コーチング開始"):
        if not json_file or not user_uploaded_file:
            st.error("JSONファイルと、あなたのスイング動画は必須です。")
            return
        
        # JSON読込
        try:
            input_data = json.load(json_file)
            basic_info = input_data.get("basic_info", {})
            coaching_policy = input_data.get("coaching_policy", {})
        except Exception as e:
            st.error(f"JSONの読み込みに失敗: {e}")
            return

        # 動画ファイルを一時保存
        user_temp_path = f"temp_{user_uploaded_file.name}"
        with open(user_temp_path, "wb") as f:
            f.write(user_uploaded_file.getbuffer())
        ideal_temp_path = None
        if ideal_uploaded_file:
            ideal_temp_path = f"temp_{ideal_uploaded_file.name}"
            with open(ideal_temp_path, "wb") as f:
                f.write(ideal_uploaded_file.getbuffer())

        # SwingCoachingSystem初期化
        config = load_config.load_config()
        system = SwingCoachingSystem(config)

        # InteractiveAgentを「streamlit」モードに切り替え
        interactive_agent = system.agents["interactive"]
        if isinstance(interactive_agent, InteractiveAgent):
            interactive_agent.mode = "streamlit"
            interactive_agent.set_streamlit_callback(get_streamlit_user_answer)

        # 非同期実行
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            system.run(
                persona_data=basic_info,
                policy_data=coaching_policy,
                user_video_path=user_temp_path,
                ideal_video_path=ideal_temp_path
            )
        )

        # 結果表示
        st.subheader("1. Interactive Q&A")
        conv = result.get("interactive_questions", [])
        if conv:
            for c in conv:
                st.write(f"- {c}")
        else:
            st.write("No conversation logs found.")

        st.subheader("2. Motion Analysis")
        st.json(result.get("motion_analysis", {}))

        st.subheader("3. Goal Setting")
        st.json(result.get("goal_setting", {}))

        st.subheader("4. Training Plan")
        st.json(result.get("training_plan", {}))

        st.subheader("5. Search Results")
        st.json(result.get("search_results", {}))

        st.subheader("6. Final Summary")
        fs = result.get("final_summary", {})
        if "summary" in fs:
            st.write(fs["summary"])

        # 一時ファイルの後始末
        if os.path.exists(user_temp_path):
            os.remove(user_temp_path)
        if ideal_temp_path and os.path.exists(ideal_temp_path):
            os.remove(ideal_temp_path)

if __name__ == "__main__":
    main()
