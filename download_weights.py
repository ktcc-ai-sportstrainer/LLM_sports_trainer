import os
import urllib.request

# ダウンロード対象と保存先パス
downloads = [
    {
        "url": "https://storage.googleapis.com/weight-llm-sport-trainer/yolov3.weights",
        "dest": "MotionAGFormer/run/lib/checkpoint/yolov3.weights",
    },
    {
        "url": "https://storage.googleapis.com/weight-llm-sport-trainer/pose_hrnet_w48_384x288.pth",
        "dest": "MotionAGFormer/run/lib/checkpoint/pose_hrnet_w48_384x288.pth",
    },
    {
        "url": "https://storage.googleapis.com/weight-llm-sport-trainer/motionagformer-b-h36m.pth.tr",
        "dest": "MotionAGFormer/checkpoint/motionagformer-b-h36m.pth.tr",
    },
]

for file in downloads:
    os.makedirs(os.path.dirname(file["dest"]), exist_ok=True)
    print(f"Downloading {file['url']} to {file['dest']}...")
    urllib.request.urlretrieve(file["url"], file["dest"])
