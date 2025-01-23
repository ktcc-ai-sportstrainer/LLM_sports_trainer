from dotenv import load_dotenv
import argparse
import json
from datetime import datetime
import os
import asyncio
from config.load_config import load_config
from core.system import SwingCoachingSystem
from utils.json_handler import JSONHandler
from langchain_openai import ChatOpenAI

# .envを読み込む (OPENAI_API_KEYなど)
load_dotenv(override=True)

required_keys = ["OPENAI_API_KEY"]
for key in required_keys:
    if not os.getenv(key):
        raise ValueError(f"{key} environment variable is not set")

def format_for_json(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

def main():
    parser = argparse.ArgumentParser(description="Swing Coaching System CLI")
    parser.add_argument("--user_video", type=str, required=True, help="Path to user's swing video file")
    parser.add_argument("--ideal_video", type=str, required=False, help="Path to ideal swing video file")
    parser.add_argument("--json", type=str, required=True, help="Path to the JSON file containing persona data")
    parser.add_argument("--interactive_cli", action="store_true", help="Enable interactive CLI mode for user Q&A")

    args = parser.parse_args()

    # JSONファイル読み込み
    try:
        input_data = JSONHandler.load_json(args.json)
        basic_info = input_data.get("basic_info", {})
        coaching_policy = input_data.get("coaching_policy", {})
    except FileNotFoundError:
        print(f"Error: JSON file not found at {args.json}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {args.json}")
        return

    # システム初期化
    config = load_config()
    system = SwingCoachingSystem(config)

    # InteractiveAgentの対話モードをCLIにするかどうか
    if args.interactive_cli:
        system.agents["interactive"].mode = "cli"
        # ↑ agent.py内の self.modeを"cli"にするように仕掛けが必要な場合がある

    # 非同期実行
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        system.run(
            persona_data=basic_info,
            policy_data=coaching_policy,
            user_video_path=args.user_video,
            ideal_video_path=args.ideal_video
        )
    )

    # 結果表示
    print("\n--- Interactive Questions & Answers ---")
    conversation = result.get("interactive_questions", [])
    if conversation:
        for entry in conversation:
            print(entry)
    else:
        print("No conversation logs found...")

    print("\n--- Motion Analysis ---")
    print(json.dumps(result.get("motion_analysis", {}), default=format_for_json, ensure_ascii=False, indent=2))

    print("\n--- Goal Setting ---")
    print(json.dumps(result.get("goal_setting", {}), default=format_for_json, ensure_ascii=False, indent=2))

    print("\n--- Training Plan ---")
    print(json.dumps(result.get("training_plan", {}), default=format_for_json, ensure_ascii=False, indent=2))

    print("\n--- Search Results ---")
    print(json.dumps(result.get("search_results", {}), default=format_for_json, ensure_ascii=False, indent=2))

    print("\n--- Final Summary ---")
    fs = result.get("final_summary", {})
    if "summary" in fs:
        print(fs["summary"])

    print("\nDone.")

if __name__ == "__main__":
    main()
