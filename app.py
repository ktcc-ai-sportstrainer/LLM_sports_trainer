# app.py

import streamlit as st
from config import load_config
from core.system import SwingCoachingSystem
import os

def main():
    st.title("野球スイングコーチングAI")

    st.sidebar.header("選手情報 (ペルソナ)")
    age = st.sidebar.number_input("年齢", min_value=5, max_value=100, value=20)
    experience = st.sidebar.text_input("野球経験", "1年")
    level = st.sidebar.selectbox("現在のレベル", ["初心者", "中級者", "上級者"])
    height = st.sidebar.number_input("身長(cm)", min_value=100, max_value=250, value=170)
    weight = st.sidebar.number_input("体重(kg)", min_value=30, max_value=150, value=60)
    position = st.sidebar.selectbox("ポジション", ["投手", "捕手", "内野手", "外野手"])
    batting_style = st.sidebar.selectbox("打席", ["右打ち", "左打ち", "両打ち"])

    st.sidebar.header("指導方針")
    focus_points = st.sidebar.multiselect(
        "重点的に指導したい項目",
        ["重心移動", "バットコントロール", "タイミング", "パワー", "コンパクトさ"],
        default=["重心移動", "バットコントロール"]
    )
    teaching_style = st.sidebar.selectbox(
        "指導スタイル",
        ["詳細な技術指導", "メンタル面重視", "基礎重視", "実践重視"]
    )
    goal = st.sidebar.text_input("達成したい目標", "安定したコンタクト")

    # 2つの動画ファイルを受け取る
    st.write("## あなたのスイング動画")
    user_uploaded_file = st.file_uploader("アップロードしてください", type=["mp4","mov","avi"])
    st.write("## 理想のスイング動画")
    ideal_uploaded_file = st.file_uploader("理想とする選手や自身の別の好調時動画など", type=["mp4","mov","avi"])

    if st.button("コーチング開始"):
        if user_uploaded_file is None or ideal_uploaded_file is None:
            st.error("あなたのスイング動画と理想のスイング動画、両方をアップロードしてください。")
            return

        # 一時ファイルに保存
        user_temp_path = f"temp_user_{user_uploaded_file.name}"
        with open(user_temp_path, "wb") as f:
            f.write(user_uploaded_file.getbuffer())

        ideal_temp_path = f"temp_ideal_{ideal_uploaded_file.name}"
        with open(ideal_temp_path, "wb") as f:
            f.write(ideal_uploaded_file.getbuffer())

        # ペルソナ情報と指導方針
        persona_data = {
            "age": age,
            "experience": experience,
            "level": level,
            "height": height,
            "weight": weight,
            "position": position,
            "batting_style": batting_style
        }
        policy_data = {
            "focus_points": focus_points,
            "teaching_style": teaching_style,
            "goal": goal
        }

        # システム実行
        config = load_config.load_config()
        system = SwingCoachingSystem(config)

        with st.spinner("分析中...少々お待ちください。"):
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(
                system.run(
                    persona_data,
                    policy_data,
                    user_video_path=user_temp_path,
                    ideal_video_path=ideal_temp_path
                )
            )

        # 結果表示
        st.success("分析が完了しました！")

        st.subheader("1. InteractiveAgentの追加質問")
        for q in result.get("interactive_questions", []):
            st.write("- " + q)

        st.subheader("2. スイング解析 (ModelingAgent)")
        st.json(result.get("motion_analysis", {}))

        st.subheader("3. 目標設定 (GoalSettingAgent)")
        st.json(result.get("goal_setting", {}))

        st.subheader("4. 練習計画 (PlanAgent)")
        st.json(result.get("training_plan", {}))

        st.subheader("5. 参考リソース (SearchAgent)")
        st.json(result.get("search_results", {}))

        st.subheader("6. 最終まとめ (SummarizeAgent)")
        final_summary = result.get("final_summary", {})
        if "summary" in final_summary:
            st.write(final_summary["summary"])

        # 後始末: 一時ファイル削除
        if os.path.exists(user_temp_path):
            os.remove(user_temp_path)
        if os.path.exists(ideal_temp_path):
            os.remove(ideal_temp_path)

if __name__ == "__main__":
    main()
