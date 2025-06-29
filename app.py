import streamlit as st
import os
import json
import asyncio
from datetime import datetime

# 環境変数の読み込み（.envなどにGOOGLE_API_KEYを設定）
from dotenv import load_dotenv
load_dotenv(override=True)

from config.load_config import load_config
from core.webui.system import WebUISwingCoachingSystem
from core.base.logger import SystemLogger
from agents.interactive_agent.agent import InteractiveAgent

def init_session_state():
    """セッション変数の初期化"""
    if "current_step" not in st.session_state:
        st.session_state.current_step = 1  # ステップ1～3
    if "user_json_path" not in st.session_state:
        st.session_state.user_json_path = None
    if "ideal_json_path" not in st.session_state:
        st.session_state.ideal_json_path = None
    if "visualization_path" not in st.session_state:
        st.session_state.visualization_path = None
    if "ideal_visualization_path" not in st.session_state:
        st.session_state.ideal_visualization_path = None
    if "pose_estimation_completed" not in st.session_state:
        st.session_state.pose_estimation_completed = False

    # Step2で生成した質問一覧
    if "generated_questions" not in st.session_state:
        st.session_state.generated_questions = []
    # Step2でのユーザー回答
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = []
    # Step2: InteractiveAgentの対話結果
    if "interactive_result" not in st.session_state:
        st.session_state.interactive_result = {}

    # Step3 全エージェント分析の最終結果
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None

def run_sync(coro):
    """非同期処理を同期的に実行するためのヘルパー関数"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

def save_temp_file(uploaded_file, prefix):
    """一時ファイルとしてアップロードを保存し、そのパスを返す"""
    if not uploaded_file:
        return None
    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{prefix}_{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def cleanup():
    """一時ファイルの削除"""
    temp_dir = "temp_files"
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))

def validate_inputs(basic_info, coaching_policy):
    """最低限のバリデーション: 名前・目標が必須"""
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

# ==================== メインアプリ ====================
def main():
    st.set_page_config(page_title="野球スイングコーチングAI", page_icon="⚾", layout="wide")
    st.title("LLM Baseball Swing Coach")

    # セッション変数を初期化
    init_session_state()

    # ロガー & システム初期化
    logger = SystemLogger()
    config = load_config()
    system = WebUISwingCoachingSystem(config)

    # sidebar入力 (基本情報 & 指導方針)
    st.sidebar.header("選手情報")
    basic_info = {
        "name": st.sidebar.text_input("名前"),
        "age": st.sidebar.number_input("年齢", min_value=6, max_value=100, value=16),
        "grade": st.sidebar.text_input("学年"),
        "position": st.sidebar.selectbox("ポジション", ["投手","捕手","内野手","外野手"]),
        "dominant_hand": {
            "batting": st.sidebar.selectbox("打席", ["右打ち","左打ち","両打ち"]),
            "throwing": st.sidebar.selectbox("投球", ["右投げ","左投げ"])
        },
        "height": st.sidebar.number_input("身長(cm)", min_value=100, max_value=220, value=170),
        "weight": st.sidebar.number_input("体重(kg)", min_value=30, max_value=150, value=60),
        "experience": {
            "years": st.sidebar.number_input("野球経験年数", min_value=0, max_value=20, value=3),
            "history": st.sidebar.text_area("野球の経歴")
        },
        "goal": st.sidebar.text_area("達成したい目標"),
        "practice_time": st.sidebar.text_input("普段の練習時間"),
        "personal_issues": [x for x in st.sidebar.text_area("現状の課題(改行区切り)").split('\n') if x.strip()]
    }

    st.sidebar.header("指導方針")
    coaching_policy = {
        "philosophy": st.sidebar.text_area("指導の基本方針"),
        "focus_points": [x for x in st.sidebar.text_area("重点指導ポイント(改行区切り)").split('\n') if x.strip()],
        "teaching_style": st.sidebar.selectbox("指導スタイル", ["基礎重視","実践重視","メンタル重視","バランス重視"]),
        "player_strengths": [x for x in st.sidebar.text_area("選手の強み(改行区切り)").split('\n') if x.strip()],
        "player_weaknesses": [x for x in st.sidebar.text_area("選手の課題(改行区切り)").split('\n') if x.strip()]
    }

    # ===== サイドバーで "解析後の動画" を常時表示 =====
    if st.session_state.visualization_path:
        st.sidebar.video(st.session_state.visualization_path)
        st.sidebar.caption("解析済みスイング動画")

    # Step2での聞き取り内容をサイドバーに表示
    if st.session_state.user_answers:
        st.sidebar.write("### Step2 聞き取り内容")
        for i, ans in enumerate(st.session_state.user_answers):
            st.sidebar.write(f"Q{i+1}: {ans}")

    # サイドバー下部に「現在のステップ」表示
    st.sidebar.write(f"■ 現在のステップ: {st.session_state.current_step}")

    # ========== STEP 1: 3D姿勢推定 ==========
    if st.session_state.current_step == 1:
        st.subheader("Step 1: 3D姿勢推定")

        st.write("以下のいずれかの方法であなたのスイングデータを入力し、3D姿勢推定を実行します。")

        # 動画 or JSON を選択
        user_input_type = st.radio(
            "スイングデータの種類",
            ["動画をアップロード", "3D姿勢データ(JSON)をアップロード"]
        )

        if user_input_type == "動画をアップロード":
            user_uploaded_file = st.file_uploader("あなたのスイング動画をアップロード", type=["mp4","mov","avi"])
            if user_uploaded_file and st.button("3D姿勢推定を実行"):
                if not validate_inputs(basic_info, coaching_policy):
                    st.stop()
                with st.spinner("推定を実行中..."):
                    user_video_path = save_temp_file(user_uploaded_file, "user_video")
                    try:
                        pose_json_path, vis_video_path, _ = run_sync(system.process_video(user_video_path))
                        st.session_state.user_json_path = pose_json_path
                        st.session_state.visualization_path = vis_video_path
                        st.session_state.pose_estimation_completed = True
                        st.success("推定が完了しました。次に進んでください。")
                    except Exception as e:
                        st.error(f"エラー: {e}")

        else:  # JSONアップロード
            user_json_file = st.file_uploader("あなたの3D姿勢データ(JSON)をアップロード", type=["json"])
            if user_json_file:
                if not validate_inputs(basic_info, coaching_policy):
                    st.stop()
                json_path = save_temp_file(user_json_file, "user_pose")
                st.session_state.user_json_path = json_path
                st.session_state.pose_estimation_completed = True
                st.success("JSONを登録しました。")

        # Optional: 理想スイング
        with st.expander("理想スイングデータ(任意)"):
            ideal_type = st.radio("理想スイングの種類", ["なし","動画をアップロード","3D姿勢JSON"])
            if ideal_type == "動画をアップロード":
                ideal_vid = st.file_uploader("理想動画をアップロード", type=["mp4","mov","avi"], key="ideal_vid")
                if ideal_vid and st.button("理想動画の3D姿勢推定を実行"):
                    ideal_temp_path = save_temp_file(ideal_vid, "ideal_video")
                    with st.spinner("理想スイング推定..."):
                        try:
                            pose_json_path, vis_video_path, _ = run_sync(system.process_video(ideal_temp_path))
                            st.session_state.ideal_json_path = pose_json_path
                            st.session_state.ideal_visualization_path = vis_video_path
                            st.success("理想スイング推定完了。")
                        except Exception as e:
                            st.error(f"エラー: {e}")
            elif ideal_type == "3D姿勢JSON":
                ideal_json = st.file_uploader("理想3D姿勢JSON", type=["json"], key="ideal_json")
                if ideal_json:
                    ideal_json_path = save_temp_file(ideal_json, "ideal_pose")
                    st.session_state.ideal_json_path = ideal_json_path
                    st.success("理想JSONを登録しました。")

        # 生成された動画のプレビュー
        if st.session_state.visualization_path:
            st.write("### 解析結果のスイング動画プレビュー")
            st.video(st.session_state.visualization_path)

        # 次のステップへ
        if st.button("Step 2へ", disabled=not st.session_state.pose_estimation_completed):
            st.session_state.current_step = 2

    # ========== STEP 2: 対話(聞き取り)のみ ==========
    elif st.session_state.current_step == 2:
        st.subheader("Step 2: インタラクティブセッション (聞き取り)")

        st.write("ペルソナ情報や指導方針を考慮した上で、まずはAIが3つの質問を生成し、それに回答してください。")

        system.interactive_enabled = True

        # 1) 質問生成(最初に一度だけ)
        if not st.session_state.generated_questions:
            # InteractiveAgent の _generate_initial_questions を利用して質問作成
            # ※ 直接呼べない場合はPublicメソッドを作る / or ここだけ簡易利用
            agent = system.agents["interactive"]
            try:
                questions = run_sync(agent._generate_initial_questions(basic_info, coaching_policy))
                # 例えば 「\n区切りで3問」みたいに出力される想定
                st.session_state.generated_questions = questions
            except Exception as e:
                st.error(f"質問生成中にエラー: {e}")

        # 2) フォーム表示
        if st.session_state.generated_questions:
            st.write("#### 以下の質問に回答してください。")
            with st.form("hearing_form"):
                user_answers = []
                for i, question in enumerate(st.session_state.generated_questions):
                    st.write(f"Q{i+1}: {question}")
                    ans = st.text_area(f"answer_{i}", key=f"hearing_q_{i}")
                    user_answers.append(ans)
                submitted = st.form_submit_button("送信")

            if submitted:
                # 3) InteractiveAgentに対話ログとしてセット
                #    ここではシンプルに: questions + user_answers を追加する
                try:
                    # run_sync() で agent.run() を呼ぶ
                    # ただ、agent.run() 内部でも _generate_initial_questions が走るので注意が必要
                    # ここでは "conversation_history" を後から付け足す形に
                    interactive_result = run_sync(system.agents["interactive"].run(
                        persona=basic_info,
                        policy=coaching_policy
                    ))
                    # conversation_historyにQ&Aを追記
                    for q, a in zip(st.session_state.generated_questions, user_answers):
                        interactive_result["conversation_history"].append(("assistant", q))
                        interactive_result["conversation_history"].append(("user", a))

                    st.session_state.interactive_result = interactive_result
                    st.session_state.user_answers = user_answers  # サイドバー用表示

                    st.success("聞き取りが完了しました。")
                except Exception as e:
                    st.error(f"対話中にエラー: {e}")

        # 対話結果があればプレビュー表示
        if st.session_state.interactive_result:
            st.write("### 対話結果プレビュー")
            conv = st.session_state.interactive_result["conversation_history"]
            if conv:
                for speaker, msg in conv:
                    st.write(f"**{speaker}**: {msg}")

        # 次のステップへ
        if st.button("Step 3へ", disabled=(not bool(st.session_state.interactive_result))):
            st.session_state.current_step = 3

    # ========== STEP 3: モーション分析, GoalSetting, Plan, Search, Summarize ==========
    elif st.session_state.current_step == 3:
        st.subheader("Step 3: 総合分析・レポート生成")

        st.write("ここでスイング分析・目標設定・プラン作成・情報検索・最終レポート作成をまとめて実行します。")

        if st.button("分析を実行"):
            if not validate_inputs(basic_info, coaching_policy):
                st.stop()

            user_json_path = st.session_state.user_json_path
            ideal_json_path = st.session_state.ideal_json_path

            # 進捗バーやステータス表示用
            status_container = st.empty()
            progress_bar = st.progress(0)
            result_container = st.container()

            with st.spinner("分析中..."):
                try:
                    # 1) ModelingAgent
                    status_container.info("モーション分析を実行中...")
                    modeling_result = run_sync(system.agents["modeling"].run(
                        user_pose_json=user_json_path,
                        ideal_pose_json=ideal_json_path
                    ))
                    progress_bar.progress(20)

                    # 2) GoalSettingAgent
                    status_container.info("目標を設定中...")
                    conversation_insights = st.session_state.interactive_result.get("interactive_insights", [])
                    goal_result = run_sync(system.agents["goal_setting"].run(
                        persona=basic_info,
                        policy=coaching_policy,
                        conversation_insights=conversation_insights,
                        motion_analysis=modeling_result.get("analysis_result","")
                    ))
                    progress_bar.progress(40)

                    # 3) PlanAgent
                    status_container.info("トレーニングプランを作成中...")
                    plan_result = run_sync(system.agents["plan"].run(
                        goal=goal_result.get("goal_setting_result",""),
                        motion_analysis=modeling_result.get("analysis_result","")
                    ))
                    progress_bar.progress(60)

                    # 4) SearchAgent
                    status_container.info("参考情報を検索中...")
                    search_result = run_sync(system.agents["search"].run(plan_result))
                    progress_bar.progress(80)

                    # 5) SummarizeAgent
                    status_container.info("最終レポートを作成中...")
                    final_summary = run_sync(system.agents["summarize"].run(
                        analysis=modeling_result.get("analysis_result",""),
                        goal=goal_result.get("goal_setting_result",""),
                        plan=plan_result
                    ))
                    progress_bar.progress(100)

                    # 結果をまとめてセッションステートに
                    st.session_state.analysis_results = {
                        "modeling": modeling_result,
                        "goal_setting": goal_result,
                        "plan": plan_result,
                        "search_results": search_result,
                        "final_summary": final_summary
                    }

                    status_container.success("分析が完了しました！")

                except Exception as e:
                    status_container.error(f"分析中にエラーが発生: {e}")
                    st.stop()

        # 分析結果があれば表示
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            st.write("### スイング分析結果")
            st.write(results["modeling"].get("analysis_result",""))
            st.write("### 目標設定")
            st.write(results["goal_setting"].get("goal_setting_result",""))
            st.write("### トレーニングプラン")
            st.write(results["plan"])
            st.write("### 参考情報")
            st.write(results["search_results"])
            st.write("### 最終レポート")
            st.markdown(results["final_summary"])

            # ダウンロードボタン
            current_time = datetime.now().strftime('%Y%m%d')
            st.download_button(
                label="コーチングレポートをダウンロード",
                data=results["final_summary"],
                file_name=f"coaching_report_{basic_info['name']}_{current_time}.txt",
                mime="text/plain"
            )

        # リセットボタン
        st.write("---")
        if st.button("最初からやり直す"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            cleanup()
            st.experimental_rerun()

# アプリ終了時のクリーンアップ
import atexit
@atexit.register
def on_session_end():
    cleanup()

if __name__ == "__main__":
    main()
