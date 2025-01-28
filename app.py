import streamlit as st
from config.load_config import load_config
from core.webui.system import WebUISwingCoachingSystem
from core.webui.state import WebUIState
from core.base.logger import SystemLogger
import os
import json
import asyncio
import subprocess
import time
from datetime import datetime
from agents.interactive_agent.agent import InteractiveAgent
from dotenv import load_dotenv

# .envファイルを確実に読み込む
load_dotenv(override=True)

# セッション状態の初期化
if 'initialization_done' not in st.session_state:
    st.session_state.initialization_done = True
    st.session_state.update({
        'pose_estimation_completed': False,
        'conversation_history': [],
        'current_question': {},
        'analysis_results': None,
        'processing_step': None,
        'error_state': None,
        'current_progress': 0
    })

def run_sync(coro):
    """非同期処理を同期的に実行するためのヘルパー関数"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def update_progress(step: str, progress: int):
    """進捗状態を更新する関数"""
    st.session_state.processing_step = step
    st.session_state.current_progress = progress
    return st.progress(progress), st.empty()

def get_streamlit_user_answer(question: str) -> str:
    """Streamlit UIでユーザーから回答を受け取る"""
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    
    question_key = f"q_{hash(question)}"
    
    # 既に回答済みの場合はその回答を返す
    if question_key in st.session_state.responses:
        return st.session_state.responses[question_key]
    
    st.write("### これまでの会話:")
    for q, a in st.session_state.conversation_history:
        st.write(f"**Q:** {q}")
        st.write(f"**A:** {a}")
        st.write("---")
    
    st.write("### 新しい質問:")
    st.write(f"**Q:** {question}")
    
    # 回答入力欄
    answer = st.text_input("あなたの回答:", key=f"input_{question_key}")
    
    # 確定ボタン
    if st.button("回答を確定", key=f"submit_{question_key}"):
        if answer.strip():  # 空でない回答の場合
            st.session_state.conversation_history.append((question, answer))
            st.session_state.responses[question_key] = answer
            st.experimental_rerun()
            return answer
    
    return ""

def save_temp_file(uploaded_file, prefix):
    """一時ファイルを保存し、パスを返す"""
    if uploaded_file is None:
        return None
    
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, f"{prefix}_{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def cleanup():
    """リソースのクリーンアップ"""
    if 'system' in locals():
        system.cleanup()
    
    temp_dir = "temp_files"
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                os.remove(file_path)
            except Exception as e:
                st.error(f"Failed to remove temp file {file_path}: {e}")

def validate_inputs(basic_info, coaching_policy):
    """入力値のバリデーション"""
    required_fields = {
        "name": "名前",
        "goal": "目標",
    }
    
    missing_fields = []
    for field, display_name in required_fields.items():
        if not basic_info.get(field):
            missing_fields.append(display_name)
            
    if missing_fields:
        st.error(f"以下の項目は必須です: {', '.join(missing_fields)}")
        return False
    return True

def show_progress_bar(text):
    """進捗バーの表示"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"{text} {i+1}%")
        time.sleep(0.01)
    
    progress_bar.empty()
    status_text.empty()

def main():
    # ロガーの初期化
    logger = SystemLogger()

    # StreamlitのUI設定
    st.set_page_config(
        page_title="野球スイングコーチングAI",
        page_icon="⚾",
        layout="wide"
    )

    st.title("野球スイングコーチングAI")

    # インタラクティブモードの選択
    interactive_mode = st.sidebar.checkbox("インタラクティブモードを有効にする", value=True)

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
        "personal_issues": [x for x in st.sidebar.text_area("現在の課題（改行区切りで複数入力可）").split('\n') if x.strip()],
    }

    # サイドバー: 指導方針
    st.sidebar.header("指導方針")

    coaching_policy = {
        "philosophy": st.sidebar.text_area("指導の基本方針"),
        "focus_points": [x for x in st.sidebar.text_area("重点的に指導したいポイント（改行区切りで複数入力可）").split('\n') if x.strip()],
        "teaching_style": st.sidebar.selectbox(
            "指導スタイル",
            ["基礎重視", "実践重視", "メンタル重視", "バランス重視"]
        ),
        "player_strengths": [x for x in st.sidebar.text_area("選手の強み（改行区切りで複数入力可）").split('\n') if x.strip()],
        "player_weaknesses": [x for x in st.sidebar.text_area("選手の課題（改行区切りで複数入力可）").split('\n') if x.strip()],
    }

    # システムの初期化
    config = load_config()
    system = WebUISwingCoachingSystem(config)
    system.interactive_enabled = interactive_mode

    # メイン画面: スイングデータのアップロード
    st.write("## スイングデータのアップロード")

    user_input_type = st.radio(
        "あなたのスイングデータの入力方法を選択してください",
        ["動画をアップロード", "3D姿勢データ(JSON)をアップロード"]
    )

    if user_input_type == "動画をアップロード":
        user_uploaded_file = st.file_uploader(
            "あなたのスイング動画をアップロードしてください（必須）", 
            type=["mp4", "mov", "avi"]
        )
        user_json_file = None
    else:
        user_json_file = st.file_uploader(
            "あなたの3D姿勢データ(JSON)をアップロードしてください（必須）", 
            type=["json"]
        )
        if user_json_file:
            user_json_path = save_temp_file(user_json_file, "user")
            st.session_state['user_json_path'] = user_json_path
            st.session_state['pose_estimation_completed'] = True
            user_uploaded_file = None

    # 理想のスイングデータ（オプション）
    with st.expander("理想のスイングデータを追加（任意）"):
        ideal_input_type = st.radio(
            "理想のスイングデータの入力方法を選択してください",
            ["動画をアップロード", "3D姿勢データ(JSON)をアップロード"]
        )
        
        if ideal_input_type == "動画をアップロード":
            ideal_uploaded_file = st.file_uploader(
                "理想のスイング動画をアップロード", 
                type=["mp4", "mov", "avi"]
            )
            ideal_json_file = None
        else:
            ideal_json_file = st.file_uploader(
                "理想の3D姿勢データ(JSON)をアップロード", 
                type=["json"]
            )
            if ideal_json_file:
                ideal_json_path = save_temp_file(ideal_json_file, "ideal")
                st.session_state['ideal_json_path'] = ideal_json_path
                ideal_uploaded_file = None

    # 3D姿勢推定の実行（動画アップロード時のみ）
    if user_input_type == "動画をアップロード" and user_uploaded_file:
        st.write("## Step 1: 3D姿勢推定")
        if st.button("3D姿勢推定を実行"):
            if not validate_inputs(basic_info, coaching_policy):
                st.stop()

            with st.spinner('3D姿勢推定を実行中...'):
                try:
                    # ユーザー動画の処理
                    user_temp_path = save_temp_file(user_uploaded_file, "user_video")
                    pose_json_path, vis_video_path, vis_json_path = run_sync(
                        system.process_video(user_temp_path)
                    )
                    
                    # 処理結果の動画表示
                    st.success("3D姿勢推定が完了しました！")
                    st.video(vis_video_path)
                    
                    # 状態の更新
                    st.session_state['user_json_path'] = pose_json_path
                    st.session_state['visualization_path'] = vis_video_path
                    st.session_state['pose_estimation_completed'] = True

                    # 理想動画の処理（存在する場合）
                    if ideal_uploaded_file:
                        ideal_temp_path = save_temp_file(ideal_uploaded_file, "ideal_video")
                        pose_json_path, vis_video_path, vis_json_path = run_sync(
                            system.process_video(ideal_temp_path)
                        )
                        
                        st.success("理想動画の3D姿勢推定が完了しました！")
                        st.video(vis_video_path)
                        
                        st.session_state['ideal_json_path'] = pose_json_path
                        st.session_state['ideal_visualization_path'] = vis_video_path

                    st.info("Step 2のコーチング分析に進むことができます。")

                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
                    logger.log_error_details(error=e, agent="system")
                    st.stop()

    # Step 2: コーチング分析の実行
    if st.button("Step 2: コーチング分析を実行", 
                disabled=not st.session_state.get('pose_estimation_completed', False)):
        if not validate_inputs(basic_info, coaching_policy):
            st.stop()

        with st.spinner('コーチング分析を実行中...'):
            try:
                user_json_path = st.session_state.get('user_json_path')
                ideal_json_path = st.session_state.get('ideal_json_path')

                progress_bar, status_text = update_progress("コーチング分析", 0)
                
                result = run_sync(system.run(
                    persona_data=basic_info,
                    policy_data=coaching_policy,
                    user_pose_json=user_json_path,
                    ideal_pose_json=ideal_json_path
                ))

                st.session_state.analysis_results = result

                # 結果の表示
                st.write("## 分析結果")

                # 対話内容の表示（インタラクティブモードが有効な場合のみ）
                if interactive_mode and result.get("interactive"):
                    st.write("### 対話分析")
                    conversation = result["interactive"].get("conversation_history", [])
                    if conversation:
                        for speaker, msg in conversation:
                            with st.chat_message(speaker.lower()):
                                st.write(msg)

                # モーション分析の表示
                st.write("### スイング分析")
                with st.expander("詳細な分析結果を表示"):
                    st.write(result.get("motion_analysis", "分析結果がありません"))

                # 目標設定の表示
                st.write("### 設定された目標")
                st.write(result.get("goal_setting", "目標設定データがありません"))

                # トレーニングプランの表示
                st.write("### トレーニングプラン")
                with st.expander("詳細なトレーニング内容を表示"):
                    st.write(result.get("training_plan", "トレーニング計画データがありません"))

                # 関連情報の表示
                st.write("### 参考情報")
                with st.expander("収集された関連情報を表示"):
                    st.write(result.get("search_results", "関連情報がありません"))

                # 最終サマリーの表示
                st.write("## 最終コーチングレポート")
                summary_text = result.get("final_summary", "")
                if summary_text:
                    st.markdown(summary_text)
                    
                    # レポートのダウンロード機能
                    current_time = datetime.now().strftime('%Y%m%d')
                    st.download_button(
                        label="コーチングレポートをダウンロード",
                        data=summary_text,
                        file_name=f"coaching_report_{basic_info['name']}_{current_time}.txt",
                        mime="text/plain"
                    )

                st.balloons()

            except Exception as e:
                st.error(f"分析中にエラーが発生しました: {str(e)}")
                logger.log_error_details(error=e, agent="system")
                st.stop()

    # 分析進捗状況の表示
    if st.session_state.get('analysis_results'):
        st.sidebar.success("✅ 分析完了")
    else:
        st.sidebar.info("📊 分析待ち")

    # フッター
    st.markdown("---")
    st.caption("Powered by SwingCoachingSystem")

    # セッション状態のリセットボタン
    if st.sidebar.button("分析をリセット"):
        if st.sidebar.button("本当にリセットしますか？"):
            for key in ['pose_estimation_completed', 'user_json_path', 'ideal_json_path', 
                        'conversation_history', 'current_question', 'analysis_results',
                        'processing_step', 'error_state', 'current_progress',
                        'visualization_path', 'ideal_visualization_path', 'responses']:
                if key in st.session_state:
                    del st.session_state[key]
            cleanup()
            st.experimental_rerun()

def on_session_end():
    """アプリケーション終了時のクリーンアップ"""
    cleanup()

# セッション終了時のクリーンアップを登録
import atexit
atexit.register(on_session_end)

if __name__ == "__main__":
    main()