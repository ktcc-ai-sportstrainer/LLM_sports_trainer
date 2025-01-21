import streamlit as st
from config import load_config
from core.system import SwingCoachingSystem

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

    st.write("## スイング動画アップロード")
    uploaded_file = st.file_uploader("スイング動画をアップロードしてください", type=["mp4", "mov", "avi"])

    if st.button("コーチング開始"):
        if uploaded_file is None:
            st.error("動画ファイルをアップロードしてください。")
            return

        # 一時ファイルに書き出し
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ペルソナ情報と指導方針をまとめる
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

        # コンフィグ読み込み & システム実行
        config = load_config()
        system = SwingCoachingSystem(config)

        with st.spinner("分析中...少々お待ちください。"):
            result = system.run(persona_data, policy_data, temp_path)

        # 結果表示
        st.success("分析が完了しました！")

        # インタラクティブエージェントからの追加質問を表示
        st.subheader("1. InteractiveAgentの追加質問")
        for q in result["interactive_questions"]:
            st.write("- " + q)

        # スイング解析の要約
        st.subheader("2. スイング解析 (ModelingAgent)")
        st.write(result["motion_analysis"])

        # 目標設定
        st.subheader("3. 目標設定 (GoalSettingAgent)")
        st.json(result["goal_setting"])

        # 練習計画
        st.subheader("4. 練習計画 (PlanAgent)")
        st.json(result["training_plan"])

        # 検索結果 (SearchAgent)
        st.subheader("5. 参考リソース (SearchAgent)")
        st.json(result["search_results"])

        # 最終サマリー (SummarizeAgent)
        st.subheader("6. 最終まとめ")
        final_summary = result["final_summary"]["summary"]
        st.write(final_summary)

        # アップロードした一時ファイルを削除（必要に応じて）
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    main()
