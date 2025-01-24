import streamlit as st
from config import load_config
from core.system import SwingCoachingSystem
from core.logger import SystemLogger
import os
import json
import asyncio
from agents.interactive_agent.agent import InteractiveAgent

# Streamlit UIのタイトル
st.title("野球スイングコーチングAI")

# サイドバーの設定
st.sidebar.header("設定")

# ログ表示エリア（修正）
log_area = st.sidebar.text_area("Logs", "", height=300)

# ロガーの初期化
logger = SystemLogger()

def update_log_area():
    """ログをStreamlitのテキストエリアに表示"""
    logs_dir = "logs"  # ログディレクトリのパスを指定
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if os.path.isfile(os.path.join(logs_dir, f))]
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), reverse=True)

        all_logs_content = ""
        for log_file in log_files:
            with open(os.path.join(logs_dir, log_file), "r") as file:
                all_logs_content += f"--- {log_file} ---\n"
                all_logs_content += file.read()
                all_logs_content += "\n\n"

        log_area.text_area("Logs", all_logs_content, height=300)
    else:
        log_area.text_area("Logs", "ログディレクトリが存在しません。", height=300)

def get_streamlit_user_answer(question: str) -> str:
    """
    Streamlitでユーザーからの回答を受け取る。
    """
    st.write(f"### {question}")
    user_input = st.text_input("回答を入力:", key=question)
    
    # 無限ループを避けるための簡易的な条件
    if st.button("回答を送信", key=f"submit_{question}"):
        return user_input
    else:
        return ""

async def main():
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
        with st.spinner('スイングを分析中...'):
            try:
                result = await system.run(
                    persona_data=basic_info,
                    policy_data=coaching_policy,
                    user_video_path=user_temp_path,
                    ideal_video_path=ideal_temp_path
                )
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
                logger.log_error_details(e, agent="system")
                update_log_area()
                return

        # 結果表示
        st.subheader("1. Interactive Q&A")
        conv = result.get("interactive_result", {}).get("conversation_history", [])
        if conv:
            for c in conv:
                speaker, msg = c
                st.write(f"- **{speaker.capitalize()}**: {msg}")
        else:
            st.write("No conversation logs found.")

        # 変更後
        st.subheader("2. Motion Analysis")
        st.write(result.get("motion_analysis", "分析結果がありません"))

        st.subheader("3. Goal Setting")
        st.write(result.get("goal_setting", "目標設定データがありません"))

        st.subheader("4. Training Plan")
        st.write(result.get("training_plan", "トレーニング計画データがありません"))

        st.subheader("5. Search Results")
        st.write(result.get("search_results", "検索結果データがありません"))

        st.subheader("6. Final Summary")
        st.write(result.get("final_summary", "最終サマリーデータがありません"))

if __name__ == "__main__":
    asyncio.run(main())