import bvh
import cProfile
import json
import numpy as np
import bvhio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyquaternion import Quaternion
import matplotlib.path as mpath


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#自分のデータ読み込み
try:
    with open('ootanifix2.bvh', 'r') as f:
        mocap = bvh.Bvh(f.read())
except FileNotFoundError:
    print("エラー: BVHファイルが見つかりません。")
    exit()
# The package allows to make modifcation on the animation data very conviniently.

hierarchy = bvhio.readAsHierarchy('ootanifix2.bvh')

bvhio.writeHierarchy("modified.bvh", hierarchy, 1/30)

hierarchy = bvhio.readAsHierarchy('modified.bvh')
# Add a root bone to the hierarchy and set itself as 'root'.
root = bvhio.Joint('Root').attach(hierarchy, keep=['position', 'rotation', 'scale'])
# this bakes the rest pos scale of 0.0254 into the positions,
# so that the scale can be reseted to 1 again.
root.applyRestposeScale(recursive=True, bakeKeyframes=True)



#理想のスイングデータ
try:
    with open('ootanifix2.bvh', 'r') as f:
        idealmocap = bvh.Bvh(f.read())
except FileNotFoundError:
    print("エラー: BVHファイルが見つかりません。")
    exit()
# The package allows to make modifcation on the animation data very conviniently.

ideal = bvhio.readAsHierarchy('ootanifix2.bvh')

bvhio.writeHierarchy("idealmodified.bvh", ideal, 1/30)

ideal = bvhio.readAsHierarchy('idealmodified.bvh')
# Add a root bone to the hierarchy and set itself as 'root'.
ideal = bvhio.Joint('Root').attach(ideal, keep=['position', 'rotation', 'scale'])
# this bakes the rest pos scale of 0.0254 into the positions,
# so that the scale can be reseted to 1 again.
ideal.applyRestposeScale(recursive=True, bakeKeyframes=True)



GPF=[]
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




class fix:
    def ratio(selftall):

            root.loadPose(0)
            
            toe = root.filter('r_toebase_JNT')[0]
            head =  root.filter('head_JNT')[0]
                  
            tallmin=toe.PositionWorld[2]
            tallmax=head.PositionWorld[2]
           
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
              
    def segment( data, point, weight, GPF):
               
       #セグメント部位の位置の取得
       joint_strat = (mocap.get_joint_index('hips_JNT')+1) * 3
       joint_end = joint_strat + 3
       hips_Position = np.array(GPF[joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('neck_JNT')+1) * 3
       joint_end = joint_strat + 3
       neck_Position = np.array(GPF[joint_strat:joint_end])
       
       joint_strat = (mocap.get_joint_index('head_JNT')+1) * 3
       joint_end = joint_strat + 3
       head_Position = np.array(GPF[joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('l_arm_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_arm_Position = np.array(GPF[joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('r_arm_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_arm_Position =np.array(GPF[joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('l_forearm_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_forearm_Position = np.array(GPF[joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('r_forearm_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_forearm_Position = np.array(GPF[joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('l_handMiddle1_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_handMiddle_Position = np.array(GPF[joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('r_handMiddle1_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_handMiddle_Position = np.array(GPF[joint_strat:joint_end])
     
       joint_strat = (mocap.get_joint_index('l_hand_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_hand_Position = np.array(GPF[joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('r_hand_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_hand_Position = np.array(GPF[joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('l_upleg_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_upleg_Position = np.array(GPF[joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('r_upleg_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_upleg_Position = np.array(GPF[joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('l_leg_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_leg_Position = np.array(GPF[joint_strat:joint_end])
      
       joint_strat = (mocap.get_joint_index('r_leg_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_leg_Position = np.array(GPF[joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('l_foot_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_foot_Position = np.array(GPF[joint_strat:joint_end])
     
       joint_strat = (mocap.get_joint_index('r_foot_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_foot_Position = np.array(GPF[joint_strat:joint_end])
            
       joint_strat = (mocap.get_joint_index('l_toebase_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_toebase_Position = np.array(GPF[joint_strat:joint_end])
                   
       joint_strat = (mocap.get_joint_index('r_toebase_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_toebase_Position = np.array(GPF[joint_strat:joint_end])

       #各部位の重心位置の計算
       head=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][0]/100)*head_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][0]/100*neck_Position
       body=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][1]/100)*neck_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][1]/100*hips_Position
       l_uparm=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][2]/100)*l_arm_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][2]/100*l_forearm_Position
       r_uparm=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][2]/100)*r_arm_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][2]/100*r_forearm_Position
       l_forearm=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][3]/100)*l_forearm_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][3]/100*l_hand_Position
       r_forearm=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][3]/100)*r_forearm_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][3]/100*r_hand_Position
       l_hand=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][4]/100)*l_hand_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][4]/100*l_handMiddle_Position
       r_hand=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][4]/100)*r_hand_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][4]/100*r_handMiddle_Position
       l_upleg=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][5]/100)*l_upleg_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][5]/100*l_leg_Position
       r_upleg=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][5]/100)*r_upleg_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][5]/100*r_leg_Position
       l_leg=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][6]/100)*l_leg_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][6]/100*l_foot_Position
       r_leg=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][6]/100)*r_leg_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][6]/100*r_foot_Position
       l_foot=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][7]/100)*l_foot_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][7]/100*l_toebase_Position
       r_foot=(1-center_of_gravity_data[data]["data"]["質量中心比(%)"][7]/100)*r_foot_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][7]/100*r_toebase_Position
       
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

    def calculate_gravity(data, frame, Pose, weight):  
        Center_of_gravitylist=[]          
        for frame_number in range(frame):
            IGP=[]
            
            Pose.loadPose(frame_number)
                #理想のスイング
            for joint, index, depth in Pose.layout():
            
                IGP.extend(joint.PositionWorld)
            
            InpactPoint=strakezone.inpact_point(Pose)

            ideal_Center_of_gravity = center_of_gravity.segment(data, InpactPoint, weight, IGP)

            Center_of_gravitylist.append(ideal_Center_of_gravity[15])
            
        return Center_of_gravitylist


class strakezone:

        def strakezone():

            root.loadPose(0)
            
            shoulder = root.filter('l_shoulder_JNT')[0]
            hips =  root.filter('hips_JNT')[0]
            keel = root.filter('l_leg_JNT')[0]
            
            sholderPz=shoulder.PositionWorld[2]
            hipsPz=hips.PositionWorld[2]
            keelPz=keel.PositionWorld[2]
        
            #ストライクゾーンの定義
            #上限は、打者の肩の上部とユニフォームのズボンの上部の中間点に引いた水平のライン
            #下限は、ひざ頭の下部のライン
            #底面は、ホームベースの五角形
         
            maxZ=(sholderPz+hipsPz)/2
            minZ=keelPz

            raitio= fix.ratio(170)
            maxY = -76.15*raitio
            minY = -119.35*raitio

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
        
        def inpact_point(root):
   
            #インパクトポイントの定義
            r_hand_joint = root.filter('r_hand_JNT')[0]
            position_world =np.array(r_hand_joint.PositionWorld)
            
            rotation_world = r_hand_joint.RotationWorld

            ratio=fix.ratio(193)
        
            offset=np.array([0,70*ratio,0])
            q1 = Quaternion(rotation_world[0],rotation_world[1],rotation_world[2],rotation_world[3])

            InpactPoint= q1.rotate(offset) + position_world

            point=np.array([InpactPoint[0], InpactPoint[1], InpactPoint[2]])
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
        
        def calculate_strakejudege(frame, Pose):
            judge_list=[]
            for frame_number in range(frame):
                Pose.loadPose(frame_number)
                Inpactpoint=strakezone.inpact_point(Pose)
                Judge=strakezone.is_inside_pentagonal_prism(Inpactpoint)
                judge_list.append(Judge)

            return judge_list
        
        def batspeed(frame, Pose):
            speed_list=[]
            for frame_number in range(frame):
                Pose.loadPose(frame_number)
                Inpactpoint=strakezone.inpact_point(Pose)
                if Inpactpoint[1]<0:
                    Pose.loadPose(frame_number+1)
                    nextInpactpoint=strakezone.inpact_point(Pose)
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
            







class Animation:
    def __init__(self):
        self.strakezone=strakezone.strakezone()
        self.ratio=fix.ratio(170)
        self.data=center_of_gravity.dataselect('man', 1.88)  


         
    def animate(self, frame_number):
        ax.cla()
        root.loadPose(frame_number)
        GP=[]

        ax.set_title(f'Joint Position (Frame {frame_number + 1})')  # フレーム番号を表示
        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])
        ax.set_zlim([-20, 180])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        #自分のスイング
        for joint, index, depth in root.layout():
        
            GP.extend(joint.PositionWorld)
        
                # フレームごとのデータを保存
        if len(GPF) <= frame_number:
            GPF.append(GP)
        else:
            GPF[frame_number] = GP


        for joint, index, depth in root.layout():
            start_index=index*3
            end_index=start_index+3
           # WorldPosition を一度で計算してリストに格納する
        WorldPositions = np.array([GP[index * 3:index * 3 + 3] for joint, index, depth in root.layout()])

    # 一度で scatter 呼び出し
        ax.scatter(WorldPositions[:, 0], WorldPositions[:, 1], WorldPositions[:, 2], c='red')

        InpactPoint=strakezone.inpact_point(root)
        
            #ax.scatter(WorldPosition[0], WorldPosition[1], WorldPosition[2], c='red') 
            #ax.text(WorldPosition[0], WorldPosition[1], WorldPosition[2], joint.Name, fontsize=5)  # ラベルを追加
        Center_of_gravity = None
    
        if frame_number < len(GPF):
            Center_of_gravity = center_of_gravity.segment(self.data, InpactPoint, 70, GP)
        
        # Center_of_gravityがNoneでない場合にのみ処理を実行
        if Center_of_gravity is not None:
             # 各重心の座標を NumPy 配列に集約
             # ここで、Center_of_gravity が (N, 3) の形状を持つと仮定
             Center_of_gravity_positions = np.array(Center_of_gravity[:-1])  # 最後の要素を除外（もしあれば）

             # 一度で scatter 呼び出し
             ax.scatter(Center_of_gravity_positions[:, 0], Center_of_gravity_positions[:, 1], Center_of_gravity_positions[:, 2], c='black')

             # 特定の重心位置 15 を green でプロット
             ax.scatter(Center_of_gravity[15][0], Center_of_gravity[15][1], Center_of_gravity[15][2], c='green')
             print(f"{frame_number}X:{Center_of_gravity[15][0]},Y:{Center_of_gravity[15][1]},Z:{Center_of_gravity[15][2]}")
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
ani = animation.FuncAnimation(fig, animation_instance.animate, frames=len(mocap.frames), interval=0.0333) # 

plt.show()

data=center_of_gravity.dataselect('man', 1.88)
gravity=center_of_gravity.calculate_gravity(data, mocap.nframes, root, 50)
print(gravity)

print('===========================================')
idealgravity=center_of_gravity.calculate_gravity(data, idealmocap.nframes, ideal, 50)
print(idealgravity)

judge=strakezone.calculate_strakejudege(mocap.nframes, root)
print(judge)

speed=strakezone.batspeed(idealmocap.nframes, ideal)
print(speed)

for i in range(35,59):   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    frame_to_save = i  # 例として50フレーム目を指定

    # animate関数を直接呼び出し
    animation_instance.animate(frame_to_save)


    # 画像を保存
    plt.savefig(f"frame_{frame_to_save}.png")  
