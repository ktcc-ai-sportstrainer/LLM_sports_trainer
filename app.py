# streamlit_app.py
import streamlit as st
from core.system import SwingCoachingSystem
from config import load_config

def main():
    st.title("野球スイングコーチAI")

    # サイドバーでペルソナ入力
    st.sidebar.header("選手情報")
    persona = {
        "age": st.sidebar.number_input("年齢", min_value=5, max_value=100, value=20),
        "experience": st.sidebar.text_input("野球経験", value="3年"),
        "level": st.sidebar.selectbox("現在のレベル", ["初心者", "中級者", "上級者"]),
        "height": st.sidebar.number_input("身長(cm)", min_value=100, max_value=250, value=170),
        "weight": st.sidebar.number_input("体重(kg)", min_value=30, max_value=150, value=60),
        "position": st.sidebar.selectbox("ポジション", ["投手", "捕手", "内野手", "外野手"]),
        "batting_style": st.sidebar.selectbox("打席", ["右打ち", "左打ち", "両打ち"]),
    }

    # メインエリアで指導方針入力
    st.header("指導方針")
    teaching_policy = {
        "focus_points": st.multiselect(
            "重点的に指導したい項目",
            ["重心移動", "バットコントロール", "タイミング", "パワー", "コンパクトさ"],
            ["重心移動", "バットコントロール"]
        ),
        "teaching_style": st.selectbox(
            "指導スタイル",
            ["詳細な技術指導", "メンタル面重視", "基礎重視", "実践重視"]
        ),
        "goal": st.text_area("達成したい目標", "安定したコンタクトと打球方向の制御")
    }

    # ファイルアップロード
    st.header("スイング動画アップロード")
    uploaded_file = st.file_uploader("動画をアップロードしてください", type=['mp4', 'mov'])

    if st.button("分析開始") and uploaded_file is not None:
        with st.spinner('動画を分析中です...'):
            try:
                # 一時ファイルとして保存
                temp_file_path = f"temp/{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # システム実行
                config = load_config()
                system = SwingCoachingSystem(config)
                feedback = system.run(persona, teaching_policy, temp_file_path)

                # 結果表示
                st.success("分析が完了しました！")
                st.header("フィードバック")
                
                # タブで結果を整理
                tab1, tab2, tab3 = st.tabs(["技術分析", "練習メニュー", "アドバイス"])
                
                with tab1:
                    st.subheader("技術的な分析")
                    st.write(feedback["technical_analysis"])
                    # 動画やグラフの表示
                    if "visualization" in feedback:
                        st.image(feedback["visualization"])
                
                with tab2:
                    st.subheader("おすすめの練習メニュー")
                    for i, menu in enumerate(feedback["practice_menu"], 1):
                        st.write(f"{i}. {menu}")
                
                with tab3:
                    st.subheader("メンタル面のアドバイス")
                    st.write(feedback["mental_advice"])
                    st.write("応援メッセージ:")
                    st.info(feedback["encouragement"])

            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
            
            finally:
                # 一時ファイルの削除
                if "temp_file_path" in locals():
                    import os
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)

if __name__ == "__main__":
    main()