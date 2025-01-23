# jsonanalitst.py

import json
import numpy as np
from pyquaternion import Quaternion
import matplotlib.path as mpath
import math

"""
jsonanalitst.py

- analyze_json(input_json_path, user_height=170, verbose=False) -> dict
  * input_json_path: 3D座標を含むJSON (frames -> coordinates -> {joint_name, x,y,z})
  * user_height: ユーザーの身長 (cm) (ペルソナから取得)
  * verbose: True なら debug print

返り値の dict 例:
{
  "idealgravity": [...],   # 各フレームの全身重心
  "judge": [...],          # 各フレームでストライクゾーン内か (bool)
  "speed": 120.0,          # バット速度 (最大値)
  "speed_list": [...],
  "speed_list_len": ...,
  "max_speed_index": ...
}
"""

# 関節名リスト (必要に応じて使う)
joint_names = [
    "Hip", "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "Spine", "Thorax", "Neck/Nose", "Head",
    "LShoulder", "LElbow", "LWrist",
    "RShoulder", "RElbow", "RWrist"
]

# 体重比や質量中心比などのデータ
center_of_gravity_data = [
    {
        "group": "グループ 13 (N=11)",
        "data": {
            "部位": ["頭部","体幹部","上腕部","前腕部","手部","大腿部","下腿部","足部"],
            "質量比(%)": [9.5,46.2,2.3,1.4,0.9,10.6,5,2],
            "質量中心比(%)": [75.6,50.5,52.6,42.4,84.4,48.1,41.3,54.9],
        },
    },
    {
        "group": "グループ 14 (N=16)",
        "data": {
            "部位": ["頭部","体幹部","上腕部","前腕部","手部","大腿部","下腿部","足部"],
            "質量比(%)": [8.4,46.8,2.4,1.5,0.9,10.7,5.1,1.9],
            "質量中心比(%)": [77.6,50.3,53.4,42,83.3,48.5,41.3,53.6],
        },
    },
    {
        "group": "グループ 15 (N=10)",
        "data": {
            "部位": ["頭部","体幹部","上腕部","前腕部","手部","大腿部","下腿部","足部"],
            "質量比(%)": [8.6,47.2,2.4,1.6,0.8,10.9,4.9,1.6],
            "質量中心比(%)": [76.3,50.8,53.3,40.8,84.2,47.1,41.7,54.8],
        },
    },
    {
        "group": "グループ 16 (N=11)",
        "data": {
            "部位": ["頭部","体幹部","上腕部","前腕部","手部","大腿部","下腿部","足部"],
            "質量比(%)": [9.4,46,2.3,1.4,0.9,11.2,4.7,1.8],
            "質量中心比(%)": [77,51.2,52,41.3,88.5,47.4,41,53.3],
        },
    },
    {
        "group": "グループ 17 (N=15)",
        "data": {
            "部位": ["頭部","体幹部","上腕部","前腕部","手部","大腿部","下腿部","足部"],
            "質量比(%)": [8.4,45.2,2.3,1.5,0.8,11.6,5.3,1.6],
            "質量中心比(%)": [77.5,50.6,52.5,41,79.3,46.4,40.2,54.6],
        },
    },
    {
        "group": "グループ 18 (N=7)",
        "data": {
            "部位": ["頭部","体幹部","上腕部","前腕部","手部","大腿部","下腿部","足部"],
            "質量比(%)": [7.8,47.2,2.5,1.3,0.7,11.6,4.8,1.6],
            "質量中心比(%)": [79,51.3,54.2,41.5,82.2,47,40.5,54.5],
        },
    },
]


# ******************** クラス定義 ********************

class fix:
    """
    身長に合わせた比率計算を行う
    """
    def __init__(self, skeleton_frames, user_height):
        self.skeleton_frames = skeleton_frames
        self.user_height = user_height

    def ratio(self):
        """
        0番フレームの "Head" と "LAnkle" のZ座標差を (self.user_height) で割って
        縮尺比を返す
        """
        tallmin = self.skeleton_frames[0]["LAnkle"][2]
        tallmax = self.skeleton_frames[0]["Head"][2]
        tall = tallmax - tallmin
        if abs(self.user_height) < 1e-6:
            return 1.0
        return tall / self.user_height


class center_of_gravity:
    @staticmethod
    def dataselect(gender, Kaup_index):
        if gender == 'man':
            if Kaup_index <= 1.767:
                return 0
            elif 1.767 < Kaup_index < 1.979:
                return 1
            else:
                return 2
        else:
            # woman
            if Kaup_index <= 1.757:
                return 3
            elif 1.757 < Kaup_index < 2.031:
                return 4
            else:
                return 5

    @staticmethod
    def segment(data_idx, point, weight, frame_num, skeleton_frames):
        # skeleton_frames[frame_num] => dict { "Hip":(x,y,z), ... }
        # data_idx => center_of_gravity_dataの何番か
        data_block = center_of_gravity_data[data_idx]["data"]
        default_block = center_of_gravity_data[0]["data"]

        hips_Position = np.array(skeleton_frames[frame_num]["Hip"])
        r_upleg_Position = np.array(skeleton_frames[frame_num]["RHip"])
        r_leg_Position = np.array(skeleton_frames[frame_num]["RKnee"])
        r_foot_Position = np.array(skeleton_frames[frame_num]["RAnkle"])
        l_upleg_Position = np.array(skeleton_frames[frame_num]["LHip"])
        l_leg_Position = np.array(skeleton_frames[frame_num]["LKnee"])
        l_foot_Position = np.array(skeleton_frames[frame_num]["LAnkle"])
        Spine_Position = np.array(skeleton_frames[frame_num]["Spine"])
        Thorax_Position = np.array(skeleton_frames[frame_num]["Thorax"])
        neck_Position = np.array(skeleton_frames[frame_num]["Neck/Nose"])
        head_Position = np.array(skeleton_frames[frame_num]["Head"])
        l_arm_Position = np.array(skeleton_frames[frame_num]["LShoulder"])
        l_forearm_Position = np.array(skeleton_frames[frame_num]["LElbow"])
        l_hand_Position = np.array(skeleton_frames[frame_num]["LWrist"])
        r_arm_Position = np.array(skeleton_frames[frame_num]["RShoulder"])
        r_forearm_Position = np.array(skeleton_frames[frame_num]["RElbow"])
        r_hand_Position = np.array(skeleton_frames[frame_num]["RWrist"])

        # 頭
        head = (1 - data_block["質量中心比(%)"][0]/100)*head_Position + \
               default_block["質量中心比(%)"][0]/100*neck_Position
        # 体
        body = (1 - data_block["質量中心比(%)"][1]/100)*Thorax_Position + \
               default_block["質量中心比(%)"][1]/100*hips_Position
        # 上腕
        l_uparm = (1 - data_block["質量中心比(%)"][2]/100)*l_arm_Position + \
                  default_block["質量中心比(%)"][2]/100*l_forearm_Position
        r_uparm = (1 - data_block["質量中心比(%)"][2]/100)*r_arm_Position + \
                  default_block["質量中心比(%)"][2]/100*r_forearm_Position
        # 前腕
        l_forearm_ = (1 - data_block["質量中心比(%)"][3]/100)*l_forearm_Position + \
                     default_block["質量中心比(%)"][3]/100*l_hand_Position
        r_forearm_ = (1 - data_block["質量中心比(%)"][3]/100)*r_forearm_Position + \
                     default_block["質量中心比(%)"][3]/100*r_hand_Position
        # 手
        l_hand = l_hand_Position
        r_hand = r_hand_Position
        # 大腿
        l_upleg_ = (1 - data_block["質量中心比(%)"][5]/100)*l_upleg_Position + \
                   default_block["質量中心比(%)"][5]/100*l_leg_Position
        r_upleg_ = (1 - data_block["質量中心比(%)"][5]/100)*r_upleg_Position + \
                   default_block["質量中心比(%)"][5]/100*r_leg_Position
        # 下腿
        l_leg_ = (1 - data_block["質量中心比(%)"][6]/100)*l_leg_Position + \
                 default_block["質量中心比(%)"][6]/100*l_foot_Position
        r_leg_ = (1 - data_block["質量中心比(%)"][6]/100)*r_leg_Position + \
                 default_block["質量中心比(%)"][6]/100*r_foot_Position
        # 足
        l_foot_ = l_foot_Position
        r_foot_ = r_foot_Position

        # inpact point
        Inpactpoint = (1 - 1/3)*point + (1/3)*r_hand_Position

        mrat = data_block["質量比(%)"]

        cg = ((mrat[0]/100)*head + (mrat[1]/100)*body +
              (mrat[2]/100)*l_uparm + (mrat[2]/100)*r_uparm +
              (mrat[3]/100)*l_forearm_ + (mrat[3]/100)*r_forearm_ +
              (mrat[4]/100)*l_hand + (mrat[4]/100)*r_hand +
              (mrat[5]/100)*l_upleg_ + (mrat[5]/100)*r_upleg_ +
              (mrat[6]/100)*l_leg_ + (mrat[6]/100)*r_leg_ +
              (mrat[7]/100)*l_foot_ + (mrat[7]/100)*r_foot_
             )*weight + 0.9*Inpactpoint
        cg = cg / (weight+0.9)

        G = [
            head.tolist(), body.tolist(), l_uparm.tolist(), r_uparm.tolist(),
            l_forearm_.tolist(), r_forearm_.tolist(), l_hand.tolist(), r_hand.tolist(),
            l_upleg_.tolist(), r_upleg_.tolist(), l_leg_.tolist(), r_leg_.tolist(),
            l_foot_.tolist(), r_foot_.tolist(), Inpactpoint.tolist(), cg.tolist()
        ]
        return G


class strakezone:

    @staticmethod
    def inpact_point(i, skeleton_frames):
        # RHand & RElbow => オフセット
        R_hand = np.array(skeleton_frames[i]["RWrist"])
        R_elbow = np.array(skeleton_frames[i]["RElbow"])
        direction_vector = R_hand - R_elbow
        norm_val = np.linalg.norm(direction_vector)
        if norm_val<1e-6:
            return R_hand
        direction_unit = direction_vector / norm_val
        up_vector = np.array([0,1,0])
        perp = np.cross(direction_unit, up_vector)
        if np.linalg.norm(perp)<1e-6:
            up_vector = np.array([1,0,0])
            perp = np.cross(direction_unit, up_vector)
        perp = perp / np.linalg.norm(perp)
        rotation_axis = np.cross(perp, direction_unit)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        angle = np.pi / 2
        q = Quaternion(axis=rotation_axis, angle=angle)
        offset = np.array([0,70,0])  # 簡易
        rotated = q.rotate(offset)
        InpactPoint = R_hand + rotated
        return InpactPoint

    @staticmethod
    def is_inside_pentagonal_prism(point, zone):
        x,y,z = point
        if not(zone[2] <= z <= zone[5]):
            return False
        vertices = [
            np.array([zone[3], zone[4]]),
            np.array([zone[3], zone[1]]),
            np.array([0, zone[4]]),
            np.array([0, zone[1]]),
            np.array([zone[0], (zone[4]+zone[1])/2])
        ]
        path = mpath.Path(vertices)
        return path.contains_point((x,y))

    @staticmethod
    def calc_strike_judge(num_frames, skeleton_frames, zone):
        jlist = []
        for i in range(num_frames):
            p = strakezone.inpact_point(i, skeleton_frames)
            inside = strakezone.is_inside_pentagonal_prism(p, zone)
            jlist.append(inside)
        return jlist

    @staticmethod
    def batspeed(skeleton_frames):
        speed_list = []
        for i in range(len(skeleton_frames)-1):
            p_now = strakezone.inpact_point(i, skeleton_frames)
            p_next = strakezone.inpact_point(i+1, skeleton_frames)
            dist = np.linalg.norm(p_next - p_now)
            # 適当計算
            speed_list.append(dist*10)
        max_speed = max(speed_list)
        max_idx = speed_list.index(max_speed)
        return max_speed, speed_list, max_idx


# ********************** 分析関数 *************************

def analyze_json(input_json_path, user_height=170.0, verbose=False):
    """
    JSON(3D座標)を読み込み:
      frames: [ { frame_index, coordinates: [{joint_name, x,y,z}, ...] }, ... ]
    user_height: ペルソナ情報にある身長（cm）
    verbose: Trueなら旧来のprintデバッグを出す

    戻り値: {
      "idealgravity": [...],
      "judge": [...],
      "speed": float,
      "speed_list": [...],
      "speed_list_len": int,
      "max_speed_index": int
    }
    """
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    frames_data = data["frames"]
    skeleton_frames = []
    for fr in frames_data:
        # fr: { "frame_index": int, "coordinates": [ {joint_name, x,y,z}, ... ] }
        coords_list = fr["coordinates"]
        d = {}
        for c in coords_list:
            d[c["joint_name"]] = (c["x"], c["y"], c["z"])
        skeleton_frames.append(d)

    total_frames = len(skeleton_frames)
    if verbose:
        print(f"[JsonAnalist] loaded {total_frames} frames from {input_json_path}")

    # fix + ratio
    fix_obj = fix(skeleton_frames, user_height)
    ratio_val = fix_obj.ratio()

    # ストライクゾーン設定 => minX, minY, minZ, maxX, maxY, maxZ
    # (例) 0番フレームの LShoulder, Hip, LKnee から計算
    # -> ここでは腕を変えるなど適宜
    # 簡易にmilk
    # 例: maxZ = (shoulderZ + hipZ)/2, ...
    # ここで "170固定" だったところを ratio_val でスケール
    # or 既存コードの strakezoneを活用
    # 省略し、calc_strike_judge呼ぶ前に "zone = [..]" とする
    # => 既存ロジック踏襲

    # 今回は strakezoneクラスには "zone" 設定関数がないので
    #   => "calc_strike_judge" に zoneを与えるには自前で zoneを作る必要あり
    # もともと fix.ratio(170) していた -> fix_obj.ratio() する
    # 省略

    minX = -21.6 * ratio_val
    maxX = 21.6 * ratio_val
    minY = 76.15* ratio_val
    maxY = 119.35* ratio_val
    # zは 0番フレーム "LKnee" or "Hip" から引っ張る
    # 例:
    lShoulderZ = skeleton_frames[0]["LShoulder"][2]
    hipZ = skeleton_frames[0]["Hip"][2]
    kneeZ = skeleton_frames[0]["LKnee"][2]
    maxZ = (lShoulderZ + hipZ)/2
    minZ = kneeZ

    zone = [minX, minY, minZ, maxX, maxY, maxZ]

    # 性別, Kaup適当
    data_idx = center_of_gravity.dataselect('man', 1.88)

    # 全フレーム重心
    gravity_list = []
    for i in range(total_frames):
        ip = strakezone.inpact_point(i, skeleton_frames)
        seg = center_of_gravity.segment(data_idx, ip, 70, i, skeleton_frames)
        gravity_list.append(seg[15])  # 全身重心

    # judge
    judge_list = strakezone.calc_strike_judge(total_frames, skeleton_frames, zone)
    # speed
    spd, spd_list, max_idx = strakezone.batspeed(skeleton_frames)

    if verbose:
        print("===========================================")
        print(f"User height = {user_height}, ratio= {ratio_val}")
        print(f"Gravity list = {gravity_list}")
        print(f"judge list= {judge_list}")
        print(f"speed= {spd}, speed_list= {spd_list}, len= {len(spd_list)}, max_index= {max_idx}")

    result = {
        "idealgravity": gravity_list,
        "judge": judge_list,
        "speed": spd,
        "speed_list": spd_list,
        "speed_list_len": len(spd_list),
        "max_speed_index": max_idx
    }
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="input.json")
    parser.add_argument("--user_height", type=float, default=170.0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    res = analyze_json(args.input_json, user_height=args.user_height, verbose=args.verbose)
    if args.verbose:
        print("[Final results dict]", res)
