import argparse
import json
import os
from config.load_config import load_config
from core.system import SwingCoachingSystem


def main():
    parser = argparse.ArgumentParser(description="Swing Coaching System CLI")
    parser.add_argument("--video", type=str, required=True, help="Path to swing video file")
    parser.add_argument("--age", type=int, default=20, help="Age of the player")
    parser.add_argument("--experience", type=str, default="1年", help="Baseball experience")
    parser.add_argument("--level", type=str, default="初心者", help="Current skill level")
    parser.add_argument("--height", type=float, default=170.0, help="Player height in cm")
    parser.add_argument("--weight", type=float, default=60.0, help="Player weight in kg")
    parser.add_argument("--position", type=str, default="内野手", help="Player position")
    parser.add_argument("--batting_style", type=str, default="右打ち", help="Batting style")

    parser.add_argument("--focus_points", nargs="+", default=["重心移動", "バットコントロール"], help="List of focus points")
    parser.add_argument("--teaching_style", type=str, default="詳細な技術指導", help="Style of teaching")
    parser.add_argument("--goal", type=str, default="安定したコンタクト", help="High-level goal")

    args = parser.parse_args()

    # ペルソナ情報と指導方針をまとめる
    persona_data = {
        "age": args.age,
        "experience": args.experience,
        "level": args.level,
        "height": args.height,
        "weight": args.weight,
        "position": args.position,
        "batting_style": args.batting_style
    }

    policy_data = {
        "focus_points": args.focus_points,
        "teaching_style": args.teaching_style,
        "goal": args.goal
    }

    # コンフィグ読み込み & システムインスタンス化
    config = load_config()  # 例: config/config.yaml などから読み込む想定
    system = SwingCoachingSystem(config)

    # 実行：全エージェントを順番に動かして最終結果を取得
    result = system.run(persona_data, policy_data, args.video)

    # コンソールに出力（今回はfinal_summaryを中心に表示する例）
    print("\n--- Interactive Questions ---")
    for q in result["interactive_questions"]:
        print(f"Q: {q}")

    print("\n--- Motion Analysis ---")
    print(result["motion_analysis"])

    print("\n--- Goal Setting ---")
    print(json.dumps(result["goal_setting"], ensure_ascii=False, indent=2))

    print("\n--- Training Plan ---")
    print(json.dumps(result["training_plan"], ensure_ascii=False, indent=2))

    print("\n--- Search Results ---")
    print(json.dumps(result["search_results"], ensure_ascii=False, indent=2))

    print("\n--- Final Summary ---")
    print(result["final_summary"]["summary"])
    print("\nDone.")

if __name__ == "__main__":
    main()
