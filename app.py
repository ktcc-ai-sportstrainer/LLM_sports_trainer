import streamlit as st
from config import load_config
from core.system import SwingCoachingSystem
from core.logger import SystemLogger
import os
import json
import asyncio
import subprocess
from agents.interactive_agent.agent import InteractiveAgent
from dotenv import load_dotenv

# .envファイルを確実に読み込む
load_dotenv(override=True)

# 非同期処理を同期的に実行するためのヘルパー関数
def run_sync(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# Streamlit UIでユーザーから回答を受け取る関数
def get_streamlit_user_answer(question: str) -> str:
    """Streamlit UIでユーザーから回答を受け取る"""
    user_answer = st.text_input(f"Assistant: {question}", key=f"qa_{hash(question)}")
    return user_answer

# Streamlit UIのタイトル
st.title("野球スイングコーチングAI")

# サイドバー: 基本情報入力
st.sidebar.header("選手情報")

basic_info = {
    "name": st.sidebar.text_input("名前"),
    "age": st.sidebar.number_input("年齢", min_value=6, max_value=100, value=16),
    "grade": st.sidebar.text_input("学年（例：高校1年）"),
    "position": st.sidebar.selectbox(
        "ポジション",
        ["投手", "捕手", "内野手", "外野手"]
    ),
    "dominant_hand": {
        "batting": st.sidebar.selectbox("打席", ["右打ち", "左打ち", "両打ち"]),
        "throwing": st.sidebar.selectbox("投球", ["右投げ", "左投げ"])
    },
    "height": st.sidebar.number_input("身長(cm)", min_value=100, max_value=220, value=170),
    "weight": st.sidebar.number_input("体重(kg)", min_value=30, max_value=150, value=60),
    "experience": {
        "years": st.sidebar.number_input("野球経験年数", min_value=0, max_value=20, value=3),
        "history": st.sidebar.text_area("経歴（例：少年野球3年→中学野球3年）")
    },
    "goal": st.sidebar.text_area("達成したい目標"),
    "practice_time": st.sidebar.text_input("普段の練習時間（例：平日2時間）"),
    "personal_issues": st.sidebar.text_area("現在の課題（改行区切りで複数入力可）").split('\n'),
}

# サイドバー: 指導方針
st.sidebar.header("指導方針")

coaching_policy = {
    "philosophy": st.sidebar.text_area("指導の基本方針"),
    "focus_points": st.sidebar.text_area("重点的に指導したいポイント（改行区切りで複数入力可）").split('\n'),
    "teaching_style": st.sidebar.selectbox(
        "指導スタイル",
        ["基礎重視", "実践重視", "メンタル重視", "バランス重視"]
    ),
    "player_strengths": st.sidebar.text_area("選手の強み（改行区切りで複数入力可）").split('\n'),
    "player_weaknesses": st.sidebar.text_area("選手の課題（改行区切りで複数入力可）").split('\n'),
}

# ロガーの初期化
logger = SystemLogger()

# メイン画面: 動画アップロード
st.write("## スイング動画")
user_uploaded_file = st.file_uploader("あなたのスイング動画をアップロードしてください（必須）", type=["mp4", "mov", "avi"])

with st.expander("理想のスイング動画を追加（任意）"):
    ideal_uploaded_file = st.file_uploader("理想のスイング動画をアップロード", type=["mp4", "mov", "avi"])
    st.caption("理想のスイング動画をアップロードすると、あなたのスイングと比較分析を行います。")

# Step 1: 3D姿勢推定
if st.button("Step 1: 3D姿勢推定を実行"):
    if not user_uploaded_file:
        st.error("あなたのスイング動画は必須です。")
        st.stop()

    # 入力値の簡易バリデーション
    if not basic_info["name"] or not basic_info["goal"]:
        st.error("名前と目標は必須項目です。")
        st.stop()

    # 一時ファイルの保存
    user_temp_path = f"temp_{user_uploaded_file.name}"
    with open(user_temp_path, "wb") as f:
        f.write(user_uploaded_file.getbuffer())

    ideal_temp_path = None
    if ideal_uploaded_file:
        ideal_temp_path = f"temp_{ideal_uploaded_file.name}"
        with open(ideal_temp_path, "wb") as f:
            f.write(ideal_uploaded_file.getbuffer())

    # 3D姿勢推定の実行
    with st.spinner('3D姿勢推定を実行中...'):
        try:
            # ユーザー動画の処理
            output_dir = "./run/output_temp"
            os.makedirs(output_dir, exist_ok=True)
            user_json_path = os.path.join(output_dir, "user_3d.json")
            
            cmd = [
                "python",
                "MotionAGFormer/run/vis.py",
                "--video", user_temp_path
            ]
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                st.error(f"ユーザー動画の姿勢推定に失敗しました: {process.stderr}")
                st.stop()
            
            # 理想動画の処理（存在する場合）
            ideal_json_path = None
            if ideal_temp_path:
                ideal_json_path = os.path.join(output_dir, "ideal_3d.json")
                cmd = [
                    "python",
                    "MotionAGFormer/run/vis.py",
                    "--video", ideal_temp_path
                ]
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                if process.returncode != 0:
                    st.error(f"理想動画の姿勢推定に失敗しました: {process.stderr}")
                    st.stop()

            # 3D姿勢推定の結果を保存
            st.session_state['pose_estimation_completed'] = True
            st.session_state['user_json_path'] = user_json_path
            st.session_state['ideal_json_path'] = ideal_json_path
            
            st.success("3D姿勢推定が完了しました！")
            st.info("Step 2のコーチング分析に進むことができます。")

            # 一時ファイルの削除
            if os.path.exists(user_temp_path):
                os.remove(user_temp_path)
            if ideal_temp_path and os.path.exists(ideal_temp_path):
                os.remove(ideal_temp_path)

        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            logger.log_error_details(error=e, agent="system")
            st.stop()

# Step 2: コーチング分析
if st.button("Step 2: コーチング分析を実行", 
            disabled=not st.session_state.get('pose_estimation_completed', False)):
    
    with st.spinner('コーチング分析を実行中...'):
        try:
            config = load_config.load_config()
            system = SwingCoachingSystem(config)
            
            # InteractiveAgentの設定
            interactive_agent = system.agents["interactive"]
            if isinstance(interactive_agent, InteractiveAgent):
                interactive_agent.mode = "streamlit"
                interactive_agent.set_streamlit_callback(get_streamlit_user_answer)

            # 非同期処理を同期的に実行
            result = run_sync(system.run(
                persona_data=basic_info,
                policy_data=coaching_policy,
                user_pose_json=st.session_state['user_json_path'],
                ideal_pose_json=st.session_state.get('ideal_json_path')
            ))

            # 結果の表示
            st.success("コーチング分析が完了しました！")
            
            # 対話履歴の表示
            st.subheader("1. Interactive Q&A")
            conv = result.get("interactive_result", {}).get("conversation_history", [])
            if conv:
                for c in conv:
                    speaker, msg = c
                    st.write(f"- **{speaker.capitalize()}**: {msg}")
            else:
                st.write("No conversation logs found.")

            # 分析結果の表示
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

        except Exception as e:
            st.error(f"コーチング分析中にエラーが発生しました: {str(e)}")
            logger.log_error_details(error=e, agent="system")
            st.stop()

# クリーンアップボタン
if st.sidebar.button("セッションをクリア"):
    # セッション状態とテンポラリファイルのクリーンアップ
    if 'pose_estimation_completed' in st.session_state:
        del st.session_state['pose_estimation_completed']
    if 'user_json_path' in st.session_state:
        try:
            os.remove(st.session_state['user_json_path'])
        except:
            pass
        del st.session_state['user_json_path']
    if 'ideal_json_path' in st.session_state:
        try:
            os.remove(st.session_state['ideal_json_path'])
        except:
            pass
        del st.session_state['ideal_json_path']
    st.experimental_rerun()

# ログ表示エリアの更新
def update_log_area():
    logs_dir = "logs"
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if os.path.isfile(os.path.join(logs_dir, f))]
        log_files.sort(key=lambda x: os.path.getmtime(os.path.join(logs_dir, x)), reverse=True)

        all_logs_content = ""
        for log_file in log_files[:5]:  # 最新の5件のみ表示
            with open(os.path.join(logs_dir, log_file), "r") as file:
                all_logs_content += f"--- {log_file} ---\n"
                all_logs_content += file.read()
                all_logs_content += "\n\n"

        st.sidebar.text_area("Latest Logs", all_logs_content, height=300)
    else:
        st.sidebar.text_area("Logs", "ログディレクトリが存在しません。", height=300)

# ログエリアの更新（オプション）
if st.sidebar.button("ログを更新"):
    update_log_area()