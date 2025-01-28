import os
import shutil
import subprocess
from typing import Optional
import cv2
from datetime import datetime

class VideoDisplay:
    def __init__(self):
        self.temp_dir = "temp_video_display"
        os.makedirs(self.temp_dir, exist_ok=True)

    def prepare_video_display(self, video_path: str) -> str:
        """
        動画をWebUI表示用に準備
        1. 動画のフォーマット確認
        2. 必要に応じて変換
        3. 表示用の一時パスを返す
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # 出力パスの生成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        display_path = os.path.join(
            self.temp_dir,
            f"display_{timestamp}.mp4"
        )

        try:
            # 入力動画の情報を取得
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            cap.release()

            # 動画の変換（必要に応じて）
            # Streamlit対応のため、H.264コーデックを使用
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-y',
                display_path
            ]

            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if process.returncode != 0:
                raise RuntimeError(f"Video conversion failed: {process.stderr.decode()}")

            return display_path

        except Exception as e:
            if os.path.exists(display_path):
                os.remove(display_path)
            raise e

    def add_visualization(self, video_path: str, json_data: dict) -> str:
        """
        動画に可視化を追加（オプション）
        """
        # 実装予定
        pass

    def cleanup(self):
        """一時ファイルの削除"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)