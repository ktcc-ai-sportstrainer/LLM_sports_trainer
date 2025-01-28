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

def get_streamlit_user_answer(question: str) -> str:
    """Streamlit UIでユーザーから回答を受け取る"""
    st.write("### 質問:")
    st.write(question)
    
    # 回答入力欄
    answer = st.text_input("あなたの回答:", key=f"input_{hash(question)}")
    
    # 確定ボタン
    submit_button = st.button("回答を確定", key=f"submit_{hash(question)}")
    
    if submit_button and answer.strip():
        # 会話履歴に追加
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        st.session_state.conversation_history.append((question, answer.strip()))
        return answer.strip()
    
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

    # InteractiveAgentのコールバック設定
    if interactive_mode:
        system.agents["interactive"].set_streamlit_callback(get_streamlit_user_answer)

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
                        st.session_state['ideal_json_path'] = pose_json_path
                        st.session_state['ideal_visualization_path'] = vis_video_path

                    st.info("Step 2のコーチング分析に進むことができます。")

                except Exception as e:
                    st.error(f"エラーが発生しました: {str(e)}")
                    logger.log_error_details(error=e, agent="system")
                    st.stop()

    # 処理結果の動画表示セクション
    if 'visualization_path' in st.session_state:
        st.write("## スイング映像")
        # ユーザーの動画表示
        st.write("### あなたのスイング")
        st.video(st.session_state['visualization_path'])
        
        # ダウンロードボタン
        with open(st.session_state['visualization_path'], 'rb') as f:
            st.download_button(
                label="解析済み動画をダウンロード",
                data=f,
                file_name="swing_analysis.mp4",
                mime="video/mp4"
            )
        
        # 理想スイング（存在する場合）
        if 'ideal_visualization_path' in st.session_state:
            st.write("### 理想のスイング")
            st.video(st.session_state['ideal_visualization_path'])
            # 理想スイングのダウンロードボタン
            with open(st.session_state['ideal_visualization_path'], 'rb') as f:
                st.download_button(
                    label="理想スイング動画をダウンロード",
                    data=f,
                    file_name="ideal_swing.mp4",
                    mime="video/mp4"
                )

    # Step 2: コーチング分析の実行
    if st.button("Step 2: コーチング分析を実行", 
                disabled=not st.session_state.get('pose_estimation_completed', False)):
        if not validate_inputs(basic_info, coaching_policy):
            st.stop()

        # 進行状況表示用のコンテナ
        status_container = st.empty()
        result_container = st.container()
        
        with st.spinner('コーチング分析を実行中...'):
            try:
                user_json_path = st.session_state.get('user_json_path')
                ideal_json_path = st.session_state.get('ideal_json_path')

                # モーション分析
                status_container.info("🔄 モーション分析を実行中...")
                modeling_result = run_sync(system.agents["modeling"].run(
                    user_pose_json=user_json_path,
                    ideal_pose_json=ideal_json_path
                ))
                with result_container:
                    st.write("### スイング分析結果")
                    st.write(modeling_result.get("analysis_result", "分析結果がありません"))

                # インタラクティブセッション
                if interactive_mode:
                    status_container.info("🗣️ インタラクティブセッションを開始...")
                    interactive_result = run_sync(system.agents["interactive"].run(
                        persona=basic_info,
                        policy=coaching_policy
                    ))
                    with result_container:
                        st.write("### 対話分析結果")
                        conversation = interactive_result.get("conversation_history", [])
                        if conversation:
                            for speaker, msg in conversation:
                                with st.chat_message(speaker.lower()):
                                    st.write(msg)

                # 目標設定
                status_container.info("🎯 目標を設定中...")
                goal_result = run_sync(system.agents["goal_setting"].run(
                    persona=basic_info,
                    policy=coaching_policy,
                    conversation_insights=interactive_result.get("interactive_insights", []) if interactive_mode else [],
                    motion_analysis=modeling_result.get("analysis_result", "")
                ))
                with result_container:
                    st.write("### 設定された目標")
                    st.write(goal_result.get("goal_setting_result", ""))

                # トレーニングプラン作成
                status_container.info("📋 トレーニングプランを作成中...")
                plan_result = run_sync(system.agents["plan"].run(
                    goal=goal_result.get("goal_setting_result", ""),
                    motion_analysis=modeling_result.get("analysis_result", "")
                ))
                with result_container:
                    st.write("### トレーニングプラン")
                    st.write(plan_result)

                # 関連情報検索
                status_container.info("🔍 関連情報を検索中...")
                search_result = run_sync(system.agents["search"].run(plan_result))
                with result_container:
                    st.write("### 参考情報")
                    st.write(search_result)

                # 最終サマリー生成
                status_container.info("📝 最終レポートを作成中...")
                final_summary = run_sync(system.agents["summarize"].run(
                    analysis=modeling_result.get("analysis_result", ""),
                    goal=goal_result.get("goal_setting_result", ""),
                    plan=plan_result
                ))
                with result_container:
                    st.write("### 最終コーチングレポート")
                    st.markdown(final_summary)
                    
                    # レポートのダウンロード機能
                    current_time = datetime.now().strftime('%Y%m%d')
                    st.download_button(
                        label="コーチングレポートをダウンロード",
                        data=final_summary,
                        file_name=f"coaching_report_{basic_info['name']}_{current_time}.txt",
                        mime="text/plain"
                    )

                # 完了表示
                status_container.success("✅ 分析が完了しました！")
                st.balloons()

                # セッション状態の更新
                st.session_state.analysis_results = {
                    "modeling": modeling_result,
                    "interactive": interactive_result if interactive_mode else None,
                    "goal_setting": goal_result,
                    "training_plan": plan_result,
                    "search_results": search_result,
                    "final_summary": final_summary
                }

            except Exception as e:
                status_container.error(f"分析中にエラーが発生しました: {str(e)}")
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
    with st.sidebar:
        if st.button("分析をリセット"):
            reset_confirm = st.button("本当にリセットしますか？")
            if reset_confirm:
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