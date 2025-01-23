import json
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.path as mpath


# JSONファイルの読み込み
with open('input.json', 'r') as f:
    data = json.load(f)

# フレーム情報の抽出
frames = data['frames']

# 関節名のリスト
joint_names = [
    "Hip", "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "Spine", "Thorax", "Neck/Nose", "Head",
    "LShoulder", "LElbow", "LWrist",
    "RShoulder", "RElbow", "RWrist"
]


center_of_gravity_data = [
    {
        "group": "グループ 13 (N=11)",
        "data": {
            "部位": ["頭部", "体幹部", "上腕部", "前腕部", "手部", "大腿部", "下腿部", "足部"],
            "質量比(%)": [9.5, 46.2, 2.3, 1.4, 0.9, 10.6, 5, 2],
            "質量中心比(%)": [75.6, 50.5, 52.6, 42.4, 84.4, 48.1, 41.3, 54.9],
        },
    },
    {
        "group": "グループ 14 (N=16)",
        "data": {
            "部位": ["頭部", "体幹部", "上腕部", "前腕部", "手部", "大腿部", "下腿部", "足部"],
            "質量比(%)": [8.4, 46.8, 2.4, 1.5, 0.9, 10.7, 5.1, 1.9],
            "質量中心比(%)": [77.6, 50.3, 53.4, 42, 83.3, 48.5, 41.3, 53.6],
        },
    },
    {
        "group": "グループ 15 (N=10)",
        "data": {
            "部位": ["頭部", "体幹部", "上腕部", "前腕部", "手部", "大腿部", "下腿部", "足部"],
            "質量比(%)": [8.6, 47.2, 2.4, 1.6, 0.8, 10.9, 4.9, 1.6],
            "質量中心比(%)": [76.3, 50.8, 53.3, 40.8, 84.2, 47.1, 41.7, 54.8],
        },
    },
    {
        "group": "グループ 16 (N=11)",
        "data": {
            "部位": ["頭部", "体幹部", "上腕部", "前腕部", "手部", "大腿部", "下腿部", "足部"],
            "質量比(%)": [9.4, 46, 2.3, 1.4, 0.9, 11.2, 4.7, 1.8],
            "質量中心比(%)": [77, 51.2, 52, 41.3, 88.5, 47.4, 41, 53.3],
        },
    },
    {
        "group": "グループ 17 (N=15)",
        "data": {
            "部位": ["頭部", "体幹部", "上腕部", "前腕部", "手部", "大腿部", "下腿部", "足部"],
            "質量比(%)": [8.4, 45.2, 2.3, 1.5, 0.8, 11.6, 5.3, 1.6],
            "質量中心比(%)": [77.5, 50.6, 52.5, 41, 79.3, 46.4, 40.2, 54.6],
        },
    },
    {
        "group": "グループ 18 (N=7)",
        "data": {
            "部位": ["頭部", "体幹部", "上腕部", "前腕部", "手部", "大腿部", "下腿部", "足部"],
            "質量比(%)": [7.8, 47.2, 2.5, 1.3, 0.7, 11.6, 4.8, 1.6],
            "質量中心比(%)": [79, 51.3, 54.2, 41.5, 82.2, 47, 40.5, 54.5],
        },
    },
]

# 各フレームの関節座標を格納
skeleton_frames = []
for frame_data in frames:
    coords = frame_data['coordinates']
    # 関節名とその座標をディクショナリに格納
    joints = {coord['joint_name']: (coord['x'], coord['y'], coord['z']) for coord in coords}
    skeleton_frames.append(joints)

print(skeleton_frames)

class fix:
    def ratio(selftall):
                
            tallmin=skeleton_frames[0]["LAnkle"][2]
            tallmax=skeleton_frames[0]["Head"][2]
           
            #print(tallmax)
            #print(tallmin)
            tall = tallmax - tallmin
            #print(tall)
            raitio = tall/selftall
            return raitio
    
class center_of_gravity:
    def __init__(self):
        self.data=center_of_gravity.dataselect('man', 1.88)       
  
    def dataselect(gender,Kaup_index):
        #データの選択
       if gender == 'man':
           if Kaup_index <= 1.767:
               return 0
           elif 1.767 < Kaup_index < 1.979:
               return 1
           else :
               return 2
       elif gender == 'woman':
           if Kaup_index <=1.757:
               return 3
           elif 1.757 < Kaup_index < 2.031:
               return 4
           else :
               return 5
              
    def segment( data, point, weight, frame_num):


        hips_Position = np.array(skeleton_frames[frame_num]["Hip"])
        r_upleg_Position = np.array(skeleton_frames[frame_num]["RHip"])
        r_leg_Position = np.array(skeleton_frames[frame_num]["RKnee"])
        r_foot_Position = np.array(skeleton_frames[frame_num]["RAnkle"])
        l_upleg_Position = np.array(skeleton_frames[frame_num]["LHip"])
        l_leg_Position= np.array(skeleton_frames[frame_num]["LKnee"])
        l_foot_Position = np.array(skeleton_frames[frame_num]["LAnkle"])
        Spine_Position = np.array(skeleton_frames[frame_num]["Spine"])
        Thorax_Position = np.array(skeleton_frames[frame_num]["Thorax"])
        neck_Position = np.array(skeleton_frames[frame_num]["Neck/Nose"]) # 注意: "/"を含むキーはそのまま使えます
        head_Position = np.array(skeleton_frames[frame_num]["Head"])
        l_arm_Position = np.array(skeleton_frames[frame_num]["LShoulder"])
        l_forearm_Position = np.array(skeleton_frames[frame_num]["LElbow"])
        l_hand_Position = np.array(skeleton_frames[frame_num]["LWrist"])
        r_arm_Position = np.array(skeleton_frames[frame_num]["RShoulder"])
        r_forearm_Position = np.array(skeleton_frames[frame_num]["RElbow"])
        r_hand_Position = np.array(skeleton_frames[frame_num]["RWrist"])
        
       #各部位の重心位置の計算
        head=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][0]/100)*head_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][0]/100*neck_Position
        body=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][1]/100)*Thorax_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][1]/100*hips_Position
        l_uparm=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][2]/100)*l_arm_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][2]/100*l_forearm_Position
        r_uparm=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][2]/100)*r_arm_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][2]/100*r_forearm_Position
        l_forearm=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][3]/100)*l_forearm_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][3]/100*l_hand_Position
        r_forearm=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][3]/100)*r_forearm_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][3]/100*r_hand_Position
        l_hand=l_hand_Position
        r_hand=r_hand_Position
        l_upleg=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][5]/100)*l_upleg_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][5]/100*l_leg_Position
        r_upleg=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][5]/100)*r_upleg_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][5]/100*r_leg_Position
        l_leg=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][6]/100)*l_leg_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][6]/100*l_foot_Position
        r_leg=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][6]/100)*r_leg_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][6]/100*r_foot_Position
        l_foot=l_foot_Position
        r_foot=r_foot_Position

        Inpactpoint=(1-1/3)*point+1/3*r_hand_Position

        center_of_gravity= ((center_of_gravity_data[data]["data"]["質量比(%)"][0]/100*head + 
                                center_of_gravity_data[data]["data"]["質量比(%)"][1]/100*body +
                                center_of_gravity_data[data]["data"]["質量比(%)"][2]/100*l_uparm +
                                center_of_gravity_data[data]["data"]["質量比(%)"][2]/100*r_uparm +
                                center_of_gravity_data[data]["data"]["質量比(%)"][3]/100*l_forearm +
                                center_of_gravity_data[data]["data"]["質量比(%)"][3]/100*r_forearm +
                                center_of_gravity_data[data]["data"]["質量比(%)"][4]/100*l_hand +
                                center_of_gravity_data[data]["data"]["質量比(%)"][4]/100*r_hand +
                                center_of_gravity_data[data]["data"]["質量比(%)"][5]/100*l_upleg +
                                center_of_gravity_data[data]["data"]["質量比(%)"][5]/100*r_upleg +
                                center_of_gravity_data[data]["data"]["質量比(%)"][6]/100*l_leg +
                                center_of_gravity_data[data]["data"]["質量比(%)"][6]/100*r_leg +
                                center_of_gravity_data[data]["data"]["質量比(%)"][7]/100*l_foot +
                                center_of_gravity_data[data]["data"]["質量比(%)"][7]/100*r_foot 
                            )*weight + 0.9*Inpactpoint)/(weight+0.9)
        head=head.tolist()
        body=body.tolist()
        l_uparm=l_uparm.tolist()
        r_uparm=r_uparm.tolist()
        l_forearm=l_forearm.tolist()
        r_forearm=r_forearm.tolist()
        l_hand=l_hand.tolist()
        r_hand=r_hand.tolist()
        l_upleg=l_upleg.tolist()
        r_upleg=r_upleg.tolist()
        l_leg=l_leg.tolist()
        r_leg=r_leg.tolist()
        l_foot=l_foot.tolist()
        r_foot=r_foot.tolist()      
        Inpactpoint=Inpactpoint.tolist()
        center_of_gravity=center_of_gravity.tolist()


        G=[head, body, l_uparm, r_uparm,
            l_forearm, r_forearm, l_hand, 
            r_hand, l_upleg, r_upleg, l_leg,
            r_leg, l_foot, r_foot, Inpactpoint, center_of_gravity]
        
        return G

    def calculate_gravity(data, frame, weight):  
        Center_of_gravitylist=[]          
        for frame_number in range(frame):
            
            InpactPoint=strakezone.inpact_point(frame_number)

            ideal_Center_of_gravity = center_of_gravity.segment(data, InpactPoint, weight, frame_number)

            Center_of_gravitylist.append(ideal_Center_of_gravity[15])
            
        return Center_of_gravitylist


class strakezone:

        def strakezone():
            
            sholderPz = skeleton_frames[0]["LShoulder"][2]
            hipsPz = skeleton_frames[0]["Hip"][2]
            keelPz = skeleton_frames[0]["LKnee"][2]
        
            #ストライクゾーンの定義
            #上限は、打者の肩の上部とユニフォームのズボンの上部の中間点に引いた水平のライン
            #下限は、ひざ頭の下部のライン
            #底面は、ホームベースの五角形
         
            maxZ=(sholderPz+hipsPz)/2
            minZ=keelPz

            raitio= fix.ratio(170)
            minY = 76.15*raitio
            maxY = 119.35*raitio

            minX = -21.6*raitio
            maxX = 21.6*raitio

            minX= float(minX)
            minY= float(minY)
            minZ= float(minZ)
            maxX= float(maxX)
            maxY= float(maxY)
            maxZ= float(maxZ)

            strakezone=[minX, minY, minZ, maxX, maxY, maxZ]

            return strakezone
        
        def inpact_point(i):
            # インパクトポイントの定義

            # 必要な関節の座標を取得
            Right_hand_Position = np.array(skeleton_frames[i]["RWrist"])
            Right_elbow_Position = np.array(skeleton_frames[i]["RElbow"])
            
            # 身長比率を取得（身長173cmとして計算）
            ratio = fix.ratio(170)
            
            # オフセットベクトルを定義
            offset = np.array([0, 70 * ratio, 0])
            
            # RElbowからRWristへの方向ベクトルを計算
            direction_vector = Right_hand_Position - Right_elbow_Position
            direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)  # 単位ベクトルに正規化

            # オフセット方向を決定するために、RElbowからRWristへのベクトルと直行するベクトルを計算
            # 一つの方法として、グローバルY軸（上方向）と方向ベクトルの外積を取る
            up_vector = np.array([0, 1, 0])  # グローバル上方向
            perpendicular_vector = np.cross(direction_unit_vector, up_vector)
            if np.linalg.norm(perpendicular_vector) == 0:
                # 方向ベクトルが上方向と平行な場合、別の基準ベクトルを使用
                up_vector = np.array([1, 0, 0])  # グローバルX軸
                perpendicular_vector = np.cross(direction_unit_vector, up_vector)
            
            perpendicular_unit_vector = perpendicular_vector / np.linalg.norm(perpendicular_vector)
            
            # 回転軸を計算（方向ベクトルと直行ベクトルの間）
            rotation_axis = np.cross(perpendicular_unit_vector, direction_unit_vector)
            rotation_axis_unit = rotation_axis / np.linalg.norm(rotation_axis)
            
            # 回転角を90度（直角）に設定
            angle = np.pi / 2  # 90度
            
            # クォータニオンを作成
            q1 = Quaternion(axis=rotation_axis_unit, angle=angle)
            
            # オフセットを回転
            rotated_offset = q1.rotate(offset)
            
            # インパクトポイントを計算
            InpactPoint = Right_hand_Position + rotated_offset
            
            point = np.array([InpactPoint[0], InpactPoint[1], InpactPoint[2]])
            return point

        def is_inside_pentagonal_prism(point):
          
          x, y, z = point
          range= strakezone.strakezone()
          # 高さの範囲内にあるかチェック
          if not (range[2]<= z <= range[5]):  # 五角柱の高さを0からheightと仮定
            return False
          
          vertices=[
            np.array([range[3], range[4],]),
            np.array([range[3], range[1],]),
            np.array([0, range[4]]),
            np.array([0, range[1]]),
            np.array([range[0],(range[4]+ range[1])/2]),
            ]
        
          path = mpath.Path(vertices)

          return(path.contains_point(point))  # True
        
        def calculate_strakejudege(frame):
            judge_list=[]
            for frame_number in range(frame):
                Inpactpoint=strakezone.inpact_point(frame_number)
                Judge=strakezone.is_inside_pentagonal_prism(Inpactpoint)
                judge_list.append(Judge)

            return judge_list
        
        def batspeed(frame):
            speed_list=[]
            for frame_number in range(len(frames) - 1):
                Inpactpoint=strakezone.inpact_point(frame_number)
                if Inpactpoint[1]>0:
                    nextInpactpoint=strakezone.inpact_point(frame_number+1)
                    ratio=fix.ratio(193)
                    speed=(nextInpactpoint-Inpactpoint)/ratio/(0.0333)
                    speed = np.linalg.norm(speed)
                    speed = speed/100000*3600
                    speed_list.append(speed)                 
                else:
                    speed_list.append(0)
            max_speed=max(speed_list)
            max_index = speed_list.index(max_speed)
            print(speed_list)
            print(len(speed_list))
            print(max_index)
            return max_speed
        

# アニメーションのセットアップ
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

class Animation:
    def __init__(self):
        self.strakezone=strakezone.strakezone()
        self.ratio=fix.ratio(170)
        self.data=center_of_gravity.dataselect('man', 1.88)  
        
    def animate(self, frame_number):
        ax.cla()
        ax.set_title(f'Joint Position (Frame {frame_number + 1})')  # フレーム番号を表示
        ax.set_xlim(-0.75, 0.75)
        ax.set_ylim(-0.75, 0.75)
        ax.set_zlim(0, 1.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        WorldPositions = np.array([skeleton_frames[frame_number][joint_name] for joint_name in joint_names])
    # 一度で scatter 呼び出し
        ax.scatter(WorldPositions[:, 0], WorldPositions[:, 1], WorldPositions[:, 2], c='red')

        InpactPoint=strakezone.inpact_point(frame_number)
        
            #ax.scatter(WorldPosition[0], WorldPosition[1], WorldPosition[2], c='red') 
            #ax.text(WorldPosition[0], WorldPosition[1], WorldPosition[2], joint.Name, fontsize=5)  # ラベルを追加
        Center_of_gravity = None
    
        if frame_number < len(skeleton_frames):
            Center_of_gravity = center_of_gravity.segment(self.data, InpactPoint, 70, frame_number)
        
        # Center_of_gravityがNoneでない場合にのみ処理を実行
        if Center_of_gravity is not None:
             # 各重心の座標を NumPy 配列に集約
             # ここで、Center_of_gravity が (N, 3) の形状を持つと仮定
             Center_of_gravity_positions = np.array(Center_of_gravity[:-1])  # 最後の要素を除外（もしあれば）

             # 一度で scatter 呼び出し
             ax.scatter(Center_of_gravity_positions[:, 0], Center_of_gravity_positions[:, 1], Center_of_gravity_positions[:, 2], c='black')

             # 特定の重心位置 15 を green でプロット
             ax.scatter(Center_of_gravity[15][0], Center_of_gravity[15][1], Center_of_gravity[15][2], c='green')
             #print(f"{frame_number}X:{Center_of_gravity[15][0]},Y:{Center_of_gravity[15][1]},Z:{Center_of_gravity[15][2]}")
        else:
                print(f"Warning: Segment data for frame {frame_number} could not be calculated.")
    
        minX = self.strakezone[0]
        minY = self.strakezone[1]
        minZ = self.strakezone[2]
        maxX = self.strakezone[3]
        maxY = self.strakezone[4]
        maxZ = self.strakezone[5]
            
        ax.scatter(maxX, maxY, maxZ, c='green') 
        ax.scatter(maxX, minY, maxZ, c='green') 
        ax.scatter(0, maxY, maxZ, c='green') 
        ax.scatter(0, minY, maxZ, c='green') 
        ax.scatter(minX,(maxY+minY)/2, maxZ, c='green') 
        ax.scatter(maxX, maxY, minZ, c='green') 
        ax.scatter(maxX, minY, minZ, c='green') 
        ax.scatter(0, maxY, minZ, c='green') 
        ax.scatter(0, minY, minZ, c='green') 
        ax.scatter(minX,(maxY+ minY)/2, minZ, c='green') 

        raitio= self.ratio
        ax.scatter(91.9*raitio,-60.95*raitio, 0, c='blue') 
        ax.scatter(-91.9*raitio,-60.95*raitio, 0, c='blue') 
        ax.scatter(-91.9*raitio,60.95*raitio, 0, c='blue') 
        ax.scatter(91.9*raitio,60.95*raitio, 0, c='blue')  
        
        ax.scatter(InpactPoint[0], InpactPoint[1],InpactPoint[2], c='black') 
        
        #print(f"{frame_number}point1: {strakezone.is_inside_pentagonal_prism(Center_of_gravity[14], self.strakezone)}")

animation_instance = Animation()  # Animationクラスのインスタンスを作成
# アニメーションの作成
ani = FuncAnimation(fig, animation_instance.animate, frames=len(skeleton_frames), interval=50)

plt.show()




print('===========================================')
data=center_of_gravity.dataselect('man', 1.88)
idealgravity=center_of_gravity.calculate_gravity(data, len(skeleton_frames),  50)
print(idealgravity)

judge=strakezone.calculate_strakejudege(len(skeleton_frames))
print(judge)

speed=strakezone.batspeed(skeleton_frames)
print(speed)

for i in range(240,243):   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    frame_to_save = i  # 例として50フレーム目を指定

     #animate関数を直接呼び出し
    animation_instance.animate(frame_to_save)


    # 画像を保存
    plt.savefig(f"frame_{frame_to_save}.png")  

