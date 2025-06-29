import cv2
import numpy as np
from typing import Tuple, Optional

class VideoProcessor:
    def __init__(self):
        pass

    @staticmethod
    def read_video(video_path: str) -> Tuple[np.ndarray, int, int]:
        """動画を読み込み、フレーム配列とサイズを返す"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return np.array(frames), frames[0].shape[1], frames[0].shape[0]

    @staticmethod
    def save_processed_video(frames: np.ndarray, output_path: str, fps: int = 30):
        """処理済みフレームを動画として保存"""
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            output_path, 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (width, height)
        )
        for frame in frames:
            writer.write(frame)
        writer.release()