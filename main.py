from dotenv import load_dotenv
import argparse
import json
from datetime import datetime
import os
import asyncio
from config.load_config import load_config
from core.system import SwingCoachingSystem
from utils.json_handler import JSONHandler #JSONHandlerクラスをインポート

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

    # JSONファイルパスの引数を追加
    parser.add_argument("--json", type=str, required=True, help="Path to the JSON file containing persona and policy data")

    args = parser.parse_args()

    # JSONファイルからデータの読み込み
    try:
        input_data = JSONHandler.load_json(args.json)
        persona_data = input_data.get("basic_info", {})
        policy_data = input_data.get("coaching_policy", {})
    except FileNotFoundError:
        print(f"Error: JSON file not found at {args.json}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {args.json}")
        return

    # コンフィグ読み込みとシステム初期化
    config = load_config()
    system = SwingCoachingSystem(config)

    # 非同期実行
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        system.run(
            persona_data=persona_data,
            policy_data=policy_data,
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