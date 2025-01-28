import streamlit as st
from typing import Any, Dict, List, Tuple
import tempfile
import os
import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

from core.system import SwingCoachingSystem
from core.state import StateManager, create_initial_state
from config.load_config import load_config
from utils.json_handler import JSONHandler
from utils.validators import DataValidator
from utils.video import VideoProcessor

# 環境変数の設定
from dotenv import load_dotenv
load_dotenv(override=True)

# 非同期処理用のExecutorを定義
executor = ThreadPoolExecutor(max_workers=1)

async def agent_executor(agent_method, *args, **kwargs):
    """エージェントの非同期実行を行う関数"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, agent_method, *args, **kwargs)

def main():
    # コンフィグのロード
    config = load_config()
    # システムの初期化
    system = SwingCoachingSystem(config)

    st.set_page_config(page_title="ベースボールスイングコーチングシステム", page_icon=":baseball:")
    st.title("ベースボールスイングコーチングシステム")

    # サイドバーの設定
    st.sidebar.header("選手情報")
    persona_data = get_persona_input()

    st.sidebar.header("コーチング方針")
    policy_data = get_policy_input()

    # 動画アップロード
    st.sidebar.header("スイング動画")
    user_video_file = st.sidebar.file_uploader("あなたのスイング動画をアップロード", type=["mp4", "mov", "avi"], key="user_video")
    ideal_video_file = st.sidebar.file_uploader("理想のスイング動画をアップロード (任意)", type=["mp4", "mov", "avi"], key="ideal_video")

    # 分析開始ボタン
    if st.sidebar.button("コーチング開始"):
        # 必須項目チェック
        if not user_video_file:
            st.sidebar.error("スイング動画をアップロードしてください。")
            return

        # セッションステートの初期化
        st.session_state.user_video_path = None
        st.session_state.ideal_video_path = None
        st.session_state.processing_done = False
        st.session_state.error_message = None
        st.session_state.conversation_history = []

        try:
            # 動画を一時ファイルに保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(user_video_file.name)[1]) as tmp_user_video:
                tmp_user_video.write(user_video_file.read())
                st.session_state.user_video_path = tmp_user_video.name
                
            if ideal_video_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(ideal_video_file.name)[1]) as tmp_ideal_video:
                    tmp_ideal_video.write(ideal_video_file.read())
                    st.session_state.ideal_video_path = tmp_ideal_video.name
            
            st.session_state.initial_state = create_initial_state(
                persona_data, policy_data, st.session_state.user_video_path, st.session_state.ideal_video_path
            )
            st.session_state.state_manager = StateManager(st.session_state.initial_state)
            st.session_state.current_step = "動画アップロード完了"
            st.session_state.processing_done = True

        except Exception as e:
            st.session_state.error_message = str(e)
            st.error(f"エラーが発生しました: {e}")

    # プログレスバーと状態表示（処理開始後のみ表示）
    if 'processing_done' in st.session_state and st.session_state.processing_done:

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text(f"処理中... {st.session_state.current_step}")

        # 各エージェントの実行状態を追跡する辞書
        if 'agent_states' not in st.session_state:
            st.session_state.agent_states = {
                "interactive": {"status": "pending", "progress": 5},
                "modeling": {"status": "pending", "progress": 25},
                "goal_setting": {"status": "pending", "progress": 45},
                "plan": {"status": "pending", "progress": 60},
                "search": {"status": "pending", "progress": 80},
                "summarize": {"status": "pending", "progress": 95},
            }

        # 各ステップの実行と進捗の更新
        agents_order = ["interactive", "modeling", "goal_setting", "plan", "search", "summarize"]

        # イベントループの取得
        loop = asyncio.get_event_loop()

        for agent_name in agents_order:
            if st.session_state.agent_states[agent_name]["status"] == "pending":
                try:
                    # 非同期処理開始
                    st.session_state.agent_states[agent_name]["status"] = "running"
                    status_text.text(f"処理中... {agent_name} エージェント")

                    # 実行するエージェントのメソッドを特定
                    if agent_name == "interactive":
                        agent_method = system.agents[agent_name].run
                        agent_args = (persona_data, policy_data)
                        agent_kwargs = {}
                    elif agent_name == "modeling":
                        agent_method = system.agents[agent_name].run
                        agent_args = ()
                        agent_kwargs = {
                            "user_video_path": st.session_state.user_video_path,
                            "ideal_video_path": st.session_state.ideal_video_path
                        }
                    elif agent_name == "goal_setting":
                        agent_method = system.agents[agent_name].run
                        agent_args = (
                            st.session_state.state_manager.state["persona_data"],
                            st.session_state.state_manager.state["policy_data"],
                            st.session_state.state_manager.state["conversation"],
                            st.session_state.state_manager.state["motion_analysis"]
                        )
                        agent_kwargs = {}
                    elif agent_name == "plan":
                        agent_method = system.agents[agent_name].run
                        agent_args = (
                            st.session_state.state_manager.state["goals"],
                            st.session_state.state_manager.state["motion_analysis"]
                        )
                        agent_kwargs = {}
                    elif agent_name == "search":
                        agent_method = system.agents[agent_name].run
                        agent_args = (
                            st.session_state.state_manager.state["search_queries"],
                        )
                        agent_kwargs = {}
                    elif agent_name == "summarize":
                        agent_method = system.agents[agent_name].run
                        agent_args = (
                            st.session_state.state_manager.state["motion_analysis"],
                            st.session_state.state_manager.state["goals"],
                            st.session_state.state_manager.state["plan"]
                        )
                        agent_kwargs = {}
                    else:
                        raise ValueError(f"Unknown agent: {agent_name}")

                    # エージェントを非同期で実行
                    agent_result = loop.run_until_complete(
                        agent_executor(agent_method, *agent_args, **agent_kwargs)
                    )
                
                    # エージェントの実行結果を状態に保存
                    st.session_state.state_manager.update_state(agent_name, {agent_name: agent_result})
                    
                    # 処理完了
                    st.session_state.agent_states[agent_name]["status"] = "completed"
                    progress_bar.progress(st.session_state.agent_states[agent_name]["progress"])

                except Exception as e:
                    st.session_state.error_message = str(e)
                    st.error(f"エラーが発生しました: {e}")
                    break  # エラーが発生したらループを抜ける
        
        # 最終結果の表示
        if st.session_state.agent_states["summarize"]["status"] == "completed":
            status_text.text("処理完了！")
            progress_bar.progress(100)
            
            st.header("最終レポート")
            st.write(st.session_state.state_manager.state["summarize"].get("summary", "サマリーが生成されませんでした。"))

            # 結果の展開
            st.subheader("会話の洞察")
            st.write(st.session_state.state_manager.state["interactive"].get("interactive_insights", "洞察がありません。"))

            st.subheader("動作分析結果")
            st.write(st.session_state.state_manager.state["modeling"].get("analysis_result", "分析結果がありません。"))

            st.subheader("目標設定")
            st.write(st.session_state.state_manager.state["goal_setting"].get("goal_setting_result", "目標設定がありません。"))

            st.subheader("トレーニング計画")
            st.write(st.session_state.state_manager.state["plan"].get("plan", "トレーニング計画がありません。"))

            st.subheader("検索結果")
            st.write(st.session_state.state_manager.state["search"].get("search_results", "検索結果がありません。"))
        
            st.balloons()

def get_persona_input() -> Dict[str, Any]:
    """
    Streamlitのサイドバーからユーザーに基本情報を入力してもらう関数
    """
    
    # 基本情報
    name = st.sidebar.text_input("名前", "山田 太郎")
    age = st.sidebar.number_input("年齢", min_value=6, max_value=100, value=16, step=1)
    grade = st.sidebar.selectbox("学年", ["小学1年生", "小学2年生", "小学3年生", "小学4年生", "小学5年生", "小学6年生", "中学1年生", "中学2年生", "中学3年生", "高校1年生", "高校2年生", "高校3年生", "大学生", "社会人"], index=10)
    position = st.sidebar.selectbox("ポジション", ["投手", "捕手", "内野手", "外野手"], index=3)
    dominant_hand = st.sidebar.selectbox("打席", ["右打ち", "左打ち", "両打ち"], index=0)
    height = st.sidebar.number_input("身長 (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
    weight = st.sidebar.number_input("体重 (kg)", min_value=20.0, max_value=200.0, value=65.0, step=0.1)
    experience_years = st.sidebar.number_input("野球経験年数", min_value=0, max_value=50, value=8, step=1)
    experience_detail = st.sidebar.text_area("所属チーム・ポジション歴など", "少年野球(6年、外野手)\n中学野球(3年、内野手)")
    goal = st.sidebar.text_input("目標", "レギュラー獲得")
    practice_time = st.sidebar.text_input("普段の練習時間", "平日2時間")
    personal_issues = st.sidebar.text_area("バッティングにおける個人的な課題", "タイミングが合わない、内角が苦手")
    additional_info = st.sidebar.text_area("その他、特記事項", "部活動以外で自主練習も行っている")

    return {
        "name": name,
        "age": age,
        "grade": grade,
        "position": position,
        "dominant_hand": {"batting": dominant_hand},
        "height": height,
        "weight": weight,
        "experience": {
            "years": experience_years,
            "history": experience_detail
        },
        "goal": goal,
        "practice_time": practice_time,
        "personal_issues": personal_issues.split('\n'),
        "additional_info": additional_info
    }

def get_policy_input() -> Dict[str, Any]:
    """
    Streamlitのサイドバーからユーザーにコーチング方針を入力してもらう関数
    """

    philosophy = st.sidebar.text_area("コーチングの基本方針", "基礎技術の徹底")
    focus_points = st.sidebar.text_area("重点的に指導したいポイント", "重心移動\nバットコントロール")
    teaching_style = st.sidebar.selectbox("指導スタイル", ["基礎重視", "実践重視", "技術的指導", "メンタル面強化"], index=0)
    player_strengths = st.sidebar.text_area("選手の強み", "練習熱心\n集中力がある")
    player_weaknesses = st.sidebar.text_area("選手の課題", "力に頼りすぎ\n焦りがち")

    return {
        "philosophy": philosophy,
        "focus_points": focus_points.split('\n'),
        "teaching_style": teaching_style,
        "player_strengths": player_strengths.split('\n'),
        "player_weaknesses": player_weaknesses.split('\n')
    }

def cleanup_temp_files():
    """一時ファイルを削除"""
    temp_dir = "temp_files"
    if os.path.exists(temp_dir):
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to remove temp file {file_path}: {e}")

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

def on_session_end():
    """セッション終了時の処理"""
    cleanup_temp_files()
    # セッション状態のクリア（必要に応じて）
    # st.session_state.clear()

if __name__ == "__main__":
    main()
    # on_session_end() # Streamlitのセッション終了を検知する方法がないため、atexitを使用