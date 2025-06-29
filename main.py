from dotenv import load_dotenv
import argparse
import json
from datetime import datetime
import os
import asyncio
from config.load_config import load_config
from core.cli.system import SwingCoachingSystem
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
    # 既存の引数
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration file")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--json", type=str, required=True, help="Path to the JSON file containing persona data")
    
    # 動画パスの代わりに3D姿勢JSONのパスを受け取るように変更
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--user_video", type=str, help="Path to user's swing video file")
    group.add_argument("--user_pose_json", type=str, help="Path to user's 3D pose JSON file")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ideal_video", type=str, help="Path to ideal swing video file")
    group.add_argument("--ideal_pose_json", type=str, help="Path to ideal 3D pose JSON file")

    args = parser.parse_args()
    print(f"YO:{args}")

    # JSONファイル読み込み
    input_data = JSONHandler.load_json(args.json)
    basic_info = input_data.get("basic_info", {})
    coaching_policy = input_data.get("coaching_policy", {})

    # システム初期化と実行
    config = load_config()
    system = SwingCoachingSystem(config)
    
    # InteractiveAgent を CLI モードで初期化
    system.agents["interactive"].mode = "cli" 
    
    result = asyncio.run(
        system.run(
            persona_data=basic_info,
            policy_data=coaching_policy,
            user_video_path=args.user_video,
            ideal_video_path=args.ideal_video,
            user_pose_json=args.user_pose_json,
            ideal_pose_json=args.ideal_pose_json
        )
    )

    # 結果表示（必要に応じて整形）
    print("\n--- Interactive Questions & Answers ---")
    conversation = result.get("interactive_result", {}).get("conversation_history", [])
    if conversation:
        for speaker, msg in conversation:
            print(f"{speaker.upper()}: {msg}")
    else:
        print("No conversation logs found...")

    print("\n--- Motion Analysis ---")
    print(result.get("motion_analysis", "分析結果がありません"))

    print("\n--- Goal Setting ---")
    print(result.get("goal_setting", "目標設定データがありません"))

    print("\n--- Training Plan ---")
    print(result.get("training_plan", "トレーニング計画データがありません"))

    print("\n--- Search Results ---")
    print(result.get("search_results", "検索結果データがありません"))

    print("\n--- Final Summary ---")
    print(result.get("final_summary", "最終サマリーデータがありません"))

    print("\nDone.")
if __name__ == "__main__":
    main()