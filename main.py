from dotenv import load_dotenv
import argparse
import json
from datetime import datetime
import os
import asyncio
from config.load_config import load_config
from core.system import SwingCoachingSystem

# .envファイルを明示的に読み込む
load_dotenv(override=True)

# APIキーの存在確認
required_keys = ["OPENAI_API_KEY", "GOOGLE_CSE_ID", "GOOGLE_API_KEY"]
for key in required_keys:
    if not os.getenv(key):
        raise ValueError(f"{key} environment variable is not set")

def format_for_json(obj):
    """JSON変換用のヘルパー関数"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

def main():
    parser = argparse.ArgumentParser(description="Swing Coaching System CLI")

    # 動画パスの引数
    parser.add_argument("--user_video", type=str, required=True, help="Path to user's swing video file")
    parser.add_argument("--ideal_video", type=str, required=True, help="Path to ideal swing video file")

    # 選手情報の引数
    parser.add_argument("--age", type=int, default=20, help="Age of the player")
    parser.add_argument("--experience", type=str, default="1年", help="Baseball experience")
    parser.add_argument("--level", type=str, default="初心者", help="Current skill level")
    parser.add_argument("--height", type=float, default=170.0, help="Player height in cm")
    parser.add_argument("--weight", type=float, default=60.0, help="Player weight in kg")
    parser.add_argument("--position", type=str, default="内野手", help="Player position")
    parser.add_argument("--batting_style", type=str, default="右打ち", help="Batting style")

    # 指導方針の引数
    parser.add_argument("--focus_points", nargs="+", default=["重心移動", "バットコントロール"], help="List of focus points")
    parser.add_argument("--teaching_style", type=str, default="詳細な技術指導", help="Style of teaching")
    parser.add_argument("--goal", type=str, default="安定したコンタクト", help="High-level goal")

    args = parser.parse_args()

    # ペルソナ情報
    persona_data = {
        "age": args.age,
        "experience": args.experience,
        "level": args.level,
        "height": args.height,
        "weight": args.weight,
        "position": args.position,
        "batting_style": args.batting_style
    }

    # 指導方針情報
    policy_data = {
        "focus_points": args.focus_points,
        "teaching_style": args.teaching_style,
        "goal": args.goal
    }

    # コンフィグ読み込みとシステム初期化
    config = load_config()
    system = SwingCoachingSystem(config)

    # 非同期実行
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        system.run(
            persona_data,
            policy_data,
            user_video_path=args.user_video,
            ideal_video_path=args.ideal_video
        )
    )

    # 結果の出力
    print("\n--- Interactive Questions ---")
    for q in result.get("interactive_questions", []):
        print(f"Q: {q}")

    print("\n--- Motion Analysis ---")
    print(json.dumps(result.get("motion_analysis", {}), 
                    default=format_for_json, 
                    ensure_ascii=False, 
                    indent=2))

    print("\n--- Goal Setting ---")
    print(json.dumps(result.get("goal_setting", {}), 
                    default=format_for_json, 
                    ensure_ascii=False, 
                    indent=2))

    print("\n--- Training Plan ---")
    print(json.dumps(result.get("training_plan", {}), 
                    default=format_for_json, 
                    ensure_ascii=False, 
                    indent=2))

    print("\n--- Search Results ---")
    print(json.dumps(result.get("search_results", {}), 
                    default=format_for_json, 
                    ensure_ascii=False, 
                    indent=2))

    print("\n--- Final Summary ---")
    if "final_summary" in result and "summary" in result["final_summary"]:
        print(result["final_summary"]["summary"])

    print("\nDone.")

if __name__ == "__main__":
    main()