import bvh
import json
import numpy as np
import bvhio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyquaternion import Quaternion
import matplotlib.path as mpath


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

try:
    with open('Baseball4.bvh', 'r') as f:
        mocap = bvh.Bvh(f.read())
except FileNotFoundError:
    print("エラー: BVHファイルが見つかりません。")
    exit()
# The package allows to make modifcation on the animation data very conviniently.


hierarchy = bvhio.readAsHierarchy('Baseball4.bvh')


bvhio.writeHierarchy("modified.bvh", hierarchy, 1/30)

hierarchy = bvhio.readAsHierarchy('modified.bvh')
# Add a root bone to the hierarchy and set itself as 'root'.
root = bvhio.Joint('Root').attach(hierarchy, keep=['position', 'rotation', 'scale'])

# this bakes the rest pos scale of 0.0254 into the positions,
# so that the scale can be reseted to 1 again.
root.applyRestposeScale(recursive=True, bakeKeyframes=True)


GPF=[]


center_of_gravity_data = [
    {
        "group": "グループ 13 (N=11)",
        "data": {
            "部位": ["頭部", "体幹部", "上腕部", "前腕部", "手部", "大腿部", "下腿部", "足部"],
            "質量比(%)": [9.5, 46.2, 2.3, 1.4, 0.9, 10.6, 5, 2],
            "質量中心比(%)": [75.6, 50.5, 52.6, 42.4, 84.4, 48.1, 41.3, 54.9],
            "回転半径比(%)": {
                "kx": [45.6, 35.1, 27.3, 28.5, 28.7, 26.9, 28.4, 16],
                "ky": [43.4, 36.2, 27, 28.4, 29.1, 26.6, 28.3, 8.2],
                "kz": [33.8, 16.1, 9, 9.7, 10.1, 9.5, 12.1, 16.1],
            },
        },
    },
    {
        "group": "グループ 14 (N=16)",
        "data": {
            "部位": ["頭部", "体幹部", "上腕部", "前腕部", "手部", "大腿部", "下腿部", "足部"],
            "質量比(%)": [8.4, 46.8, 2.4, 1.5, 0.9, 10.7, 5.1, 1.9],
            "質量中心比(%)": [77.6, 50.3, 53.4, 42, 83.3, 48.5, 41.3, 53.6],
            "回転半径比(%)": {
                "kx": [46.8, 34.6, 27, 28.1, 29, 27.2, 28, 17],
                "ky": [45, 35.7, 26.6, 27.7, 29.5, 26.9, 27.8, 8.7],
                "kz": [34.1, 16.7, 9.5, 10.6, 9.9, 10, 12.9, 17.1],
            },
        },
    },
    {
        "group": "グループ 15 (N=10)",
        "data": {
            "部位": ["頭部", "体幹部", "上腕部", "前腕部", "手部", "大腿部", "下腿部", "足部"],
            "質量比(%)": [8.6, 47.2, 2.4, 1.6, 0.8, 10.9, 4.9, 1.6],
            "質量中心比(%)": [76.3, 50.8, 53.3, 40.8, 84.2, 47.1, 41.7, 54.8],
            "回転半径比(%)": {
                "kx": [46.4, 34.6, 28, 26.1, 30.1, 27.1, 28.1, 15.9],
                "ky": [44.6, 35.8, 25.4, 27.7, 30.7, 26.7, 28, 8.3],
                "kz": [35, 17.5, 10.7, 11.1, 10.7, 10.5, 14, 16.2],
            },
        },
    },
    {
        "group": "グループ 16 (N=11)",
        "data": {
            "部位": ["頭部", "体幹部", "上腕部", "前腕部", "手部", "大腿部", "下腿部", "足部"],
            "質量比(%)": [9.4, 46, 2.3, 1.4, 0.9, 11.2, 4.7, 1.8],
            "質量中心比(%)": [77, 51.2, 52, 41.3, 88.5, 47.4, 41, 53.3],
            "回転半径比(%)": {
                "kx": [46.5, 34.9, 28.6, 26.5, 29.1, 27.3, 27.5, 16.7],
                "ky": [44.4, 36, 26.1, 28.5, 29.5, 27, 27.5, 8.4],
                "kz": [35.6, 16.5, 9.4, 10.4, 9.5, 9.8, 12.5, 16.4],
            },
        },
    },
    {
        "group": "グループ 17 (N=15)",
        "data": {
            "部位": ["頭部", "体幹部", "上腕部", "前腕部", "手部", "大腿部", "下腿部", "足部"],
            "質量比(%)": [8.4, 45.2, 2.3, 1.5, 0.8, 11.6, 5.3, 1.6],
            "質量中心比(%)": [77.5, 50.6, 52.5, 41, 79.3, 46.4, 40.2, 54.6],
            "回転半径比(%)": {
                "kx": [47.2, 34.6, 27.7, 26.7, 27.2, 27.4, 27.2, 16],
                "ky": [44.3, 35.6, 27.6, 26.2, 27.6, 27.3, 26.8, 8.1],
                "kz": [37.7, 16.8, 10.8, 10, 8.8, 13.8, 10.2, 15.7],
            },
        },
    },
    {
        "group": "グループ 18 (N=7)",
        "data": {
            "部位": ["頭部", "体幹部", "上腕部", "前腕部", "手部", "大腿部", "下腿部", "足部"],
            "質量比(%)": [7.8, 47.2, 2.5, 1.3, 0.7, 11.6, 4.8, 1.6],
            "質量中心比(%)": [79, 51.3, 54.2, 41.5, 82.2, 47, 40.5, 54.5],
            "回転半径比(%)": {
                "kx": [48.2, 34.5, 28.1, 26.9, 27.7, 27.6, 27.7, 15.6],
                "ky": [44.4, 36, 26.1, 28.5, 29.5, 27, 27.5, 8.4],
                "kz": [35.6, 16.5, 9.4, 10.4, 9.5, 12.5, 9.8, 16.4],
            },
        },
    },
]

root.printTree()


class strakezone:

    def is_inside_pentagonal_prism(point, vertices, MINZ, MAXZ):
    

        x, y, z = point

        # 高さの範囲内にあるかチェック
        if not (MINZ<= z <= MAXZ):  # 五角柱の高さを0からheightと仮定
            return False
        
        path = mpath.Path(vertices)

        return(path.contains_point(point))  # True
    
    def segment(frame):
       
       #セグメント部位の位置の取得
       joint_strat = (mocap.get_joint_index('hips_JNT')+1) * 3
       joint_end = joint_strat + 3
       hips_Position = np.array(GPF[frame][joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('neck_JNT')+1) * 3
       joint_end = joint_strat + 3
       neck_Position = np.array(GPF[frame][joint_strat:joint_end])
       
       joint_strat = (mocap.get_joint_index('head_JNT')+1) * 3
       joint_end = joint_strat + 3
       head_Position = np.array(GPF[frame][joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('l_arm_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_arm_Position = np.array(GPF[frame][joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('r_arm_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_arm_Position =np.array(GPF[frame][joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('l_forearm_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_forearm_Position = np.array(GPF[frame][joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('r_forearm_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_forearm_Position = np.array(GPF[frame][joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('l_handMiddle1_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_handMiddle_Position = np.array(GPF[frame][joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('r_handMiddle1_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_handMiddle_Position = np.array(GPF[frame][joint_strat:joint_end])
     
       joint_strat = (mocap.get_joint_index('l_hand_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_hand_Position = np.array(GPF[frame][joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('r_hand_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_hand_Position = np.array(GPF[frame][joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('l_upleg_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_upleg_Position = np.array(GPF[frame][joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('r_upleg_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_upleg_Position = np.array(GPF[frame][joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('l_leg_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_leg_Position = np.array(GPF[frame][joint_strat:joint_end])
      
       joint_strat = (mocap.get_joint_index('r_leg_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_leg_Position = np.array(GPF[frame][joint_strat:joint_end])

       joint_strat = (mocap.get_joint_index('l_foot_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_foot_Position = np.array(GPF[frame][joint_strat:joint_end])
     
       joint_strat = (mocap.get_joint_index('r_foot_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_foot_Position = np.array(GPF[frame][joint_strat:joint_end])
            
       joint_strat = (mocap.get_joint_index('l_toebase_JNT')+1) * 3
       joint_end = joint_strat + 3
       l_toebase_Position = np.array(GPF[frame][joint_strat:joint_end])
                   
       joint_strat = (mocap.get_joint_index('r_toebase_JNT')+1) * 3
       joint_end = joint_strat + 3
       r_toebase_Position = np.array(GPF[frame][joint_strat:joint_end])

       
       toeZ = (mocap.get_joint_index('r_toebase_JNT')+1)*3
        #print(toeZ)
       toeZ = toeZ + 2
       headZ = (mocap.get_joint_index('head_JNT')+1)*3
       headZ = headZ + 2
       tallmax = GPF[0][headZ]
       tallmin = GPF[0][toeZ]
        #print(tallmax)
        #print(tallmin)
       tall = tallmax - tallmin
        #print(tall)
       raitio = tall/170

       r_hand_joint = root.filter("r_hand_JNT")[0]        
       position_world =np.array(r_hand_joint.PositionWorld)       
       rotation_world = r_hand_joint.RotationWorld 
       offset=np.array([-84*raitio*2/3,0,0])
       q1 = Quaternion(rotation_world[0],rotation_world[1],rotation_world[2],rotation_world[3])
       InpactPoint= q1.rotate(offset) + position_world
       point=np.array([InpactPoint[0], InpactPoint[1], InpactPoint[2]])

       #各部位の重心位置の計算
       head=(1-center_of_gravity_data[0]["data"]["質量中心比(%)"][0]/100)*head_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][0]/100*neck_Position
       body=(1-center_of_gravity_data[0]["data"]["質量中心比(%)"][1]/100)*neck_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][1]/100*hips_Position
       l_uparm=(1-center_of_gravity_data[0]["data"]["質量中心比(%)"][2]/100)*l_arm_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][2]/100*l_forearm_Position
       r_uparm=(1-center_of_gravity_data[0]["data"]["質量中心比(%)"][2]/100)*l_forearm_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][2]/100*l_hand_Position
       l_forearm=(1-center_of_gravity_data[0]["data"]["質量中心比(%)"][3]/100)*l_forearm_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][3]/100*l_hand_Position
       r_forearm=(1-center_of_gravity_data[0]["data"]["質量中心比(%)"][3]/100)*r_forearm_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][3]/100*r_hand_Position
       l_hand=(1-center_of_gravity_data[0]["data"]["質量中心比(%)"][4]/100)*l_hand_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][4]/100*l_handMiddle_Position
       r_hand=(1-center_of_gravity_data[0]["data"]["質量中心比(%)"][4]/100)*r_hand_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][4]/100*r_handMiddle_Position
       l_upleg=(1-center_of_gravity_data[0]["data"]["質量中心比(%)"][5]/100)*l_upleg_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][5]/100*l_leg_Position
       r_upleg=(1-center_of_gravity_data[0]["data"]["質量中心比(%)"][5]/100)*r_upleg_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][5]/100*r_leg_Position
       l_leg=(1-center_of_gravity_data[0]["data"]["質量中心比(%)"][6]/100)*l_leg_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][6]/100*l_foot_Position
       r_leg=(1-center_of_gravity_data[0]["data"]["質量中心比(%)"][6]/100)*r_leg_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][6]/100*r_foot_Position
       l_foot=(1-center_of_gravity_data[0]["data"]["質量中心比(%)"][7]/100)*l_foot_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][7]/100*l_toebase_Position
       r_foot=(1-center_of_gravity_data[0]["data"]["質量中心比(%)"][7]/100)*r_foot_Position+center_of_gravity_data[0]["data"]["質量中心比(%)"][7]/100*r_toebase_Position

       center_of_gravity=(center_of_gravity_data[0]["data"]["質量比(%)"][0]/100*head + 
                          center_of_gravity_data[0]["data"]["質量比(%)"][1]/100*body +
                          center_of_gravity_data[0]["data"]["質量比(%)"][2]/100*l_uparm +
                          center_of_gravity_data[0]["data"]["質量比(%)"][2]/100*r_uparm +
                          center_of_gravity_data[0]["data"]["質量比(%)"][3]/100*l_forearm +
                          center_of_gravity_data[0]["data"]["質量比(%)"][3]/100*r_forearm +
                          center_of_gravity_data[0]["data"]["質量比(%)"][4]/100*l_hand +
                          center_of_gravity_data[0]["data"]["質量比(%)"][4]/100*r_hand +
                          center_of_gravity_data[0]["data"]["質量比(%)"][5]/100*l_upleg +
                          center_of_gravity_data[0]["data"]["質量比(%)"][5]/100*r_upleg +
                          center_of_gravity_data[0]["data"]["質量比(%)"][6]/100*l_leg +
                          center_of_gravity_data[0]["data"]["質量比(%)"][6]/100*r_leg +
                          center_of_gravity_data[0]["data"]["質量比(%)"][7]/100*l_foot +
                          center_of_gravity_data[0]["data"]["質量比(%)"][7]/100*r_foot )/1
       
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
       center_of_gravity=center_of_gravity.tolist()

       G=[head, body, l_uparm, r_uparm,
          l_forearm, r_forearm, l_hand, 
          r_hand, l_upleg, r_upleg, l_leg,
          r_leg, l_foot, r_foot, center_of_gravity]
       
       return G





    def animate(frame_number):
        ax.clear()
        root.loadPose(frame_number)
        GP=[]

        for joint, index, depth in root.layout():
        
            GP.extend(joint.PositionWorld)
        
        
        GPF.append(GP)


        for joint, index, depth in root.layout():
            start_index=index*3
            end_index=start_index+3
            WorldPosition=GP[start_index:end_index]


            ax.scatter(WorldPosition[0], WorldPosition[1], WorldPosition[2], c='red') 
            #ax.text(WorldPosition[0], WorldPosition[1], WorldPosition[2], joint.Name, fontsize=5)  # ラベルを追加
                        

            ax.set_xlim([-100, 100])
            ax.set_ylim([-100, 100]) 
            ax.set_zlim([-20, 180])

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Joint Position (Frame {frame_number + 1})')  # フレーム番号を表示

        #ストライクゾーンの定義
        #上限は、打者の肩の上部とユニフォームのズボンの上部の中間点に引いた水平のライン
        #下限は、ひざ頭の下部のライン
        #底面は、ホームベースの五角形﻿

        sholderPz=(mocap.get_joint_index('l_shoulder_JNT')+1)*3
        sholderPz = sholderPz + 2
        hipsPz=(mocap.get_joint_index('hips_JNT')+1)*3
        hipsPz=hipsPz+2
        keelPz=(mocap.get_joint_index('l_leg_JNT')+2)*3
        keelPz=keelPz+2
        sholderPz = GPF[0][sholderPz]
        hipsPz= GPF[0][hipsPz]
        keelPz= GPF[0][keelPz]
        maxZ=(sholderPz+hipsPz)/2
        minZ=keelPz


        toeZ = (mocap.get_joint_index('r_toebase_JNT')+1)*3
        #print(toeZ)
        toeZ = toeZ + 2
        headZ = (mocap.get_joint_index('head_JNT')+1)*3
        headZ = headZ + 2
        tallmax = GPF[0][headZ]
        tallmin = GPF[0][toeZ]
        #print(tallmax)
        #print(tallmin)
        tall = tallmax - tallmin
        #print(tall)
        raitio = tall/170
    
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
        ax.scatter(91.9*raitio,-60.95*raitio, 0, c='blue') 
        ax.scatter(-91.9*raitio,-60.95*raitio, 0, c='blue') 
        ax.scatter(-91.9*raitio,60.95*raitio, 0, c='blue') 
        ax.scatter(91.9*raitio,60.95*raitio, 0, c='blue') 
        Center_of_gravity=strakezone.segment(frame_number)
        
        for i in range(len(Center_of_gravity)-1):
            ax.scatter(Center_of_gravity[i][0],Center_of_gravity[i][1],Center_of_gravity[i][2], c='black') 


        ax.scatter(Center_of_gravity[14][0],Center_of_gravity[14][1],Center_of_gravity[14][2], c='green') 


        


        vertices = [
        np.array([maxX, maxY]),
        np.array([maxX, minY]),
        np.array([0, maxY]),
        np.array([0, minY]),
        np.array([minX,(maxY+ minY)/2]),
        ]
    

        r_hand_joint = root.filter("r_hand_JNT")[0]
        position_world =np.array(r_hand_joint.PositionWorld)
        
        rotation_world = r_hand_joint.RotationWorld
    
        offset=np.array([-84*raitio,0,0])
        q1 = Quaternion(rotation_world[0],rotation_world[1],rotation_world[2],rotation_world[3])

        InpactPoint= q1.rotate(offset) + position_world

    

        point=np.array([InpactPoint[0], InpactPoint[1], InpactPoint[2]])
        ax.scatter(InpactPoint[0],InpactPoint[1],InpactPoint[2], c='black') 


        print(f"{frame_number}point1: {strakezone.is_inside_pentagonal_prism(point, vertices, minZ, maxZ )}")


ani = animation.FuncAnimation(fig, strakezone.animate, frames=113, interval=0.0333) # 

plt.show()



# 3. データを構造化する

structured_data = []
structured_data1 = []
joint_names = [] # joint_names から joint_indices に変更

for joint, index, depth in root.layout():
    joint_names.append(joint.Name) # ジョイントのインデックスを保存

for frame_index, frame_data in enumerate(GPF):
    frame_dict = {
        "frame": frame_index + 1,
        "joints": {}
    }
    # ジョイントごとにx, y, z座標を取得
    for i, joint_name in enumerate(joint_names):
        # x, y, zのインデックスを計算
        idx = i * 3
        x = frame_data[idx]
        y = frame_data[idx + 1]
        z = frame_data[idx + 2]
        # ジョイントの座標を追加
        frame_dict["joints"][joint_name] = {"x": x, "y": y, "z": z}
    # 構造化データにフレームを追加
    structured_data.append(frame_dict)
   

# ジョイントごとにx, y, z座標を取得
for i, joint_name in enumerate(joint_names):      
        frame_dict = {
                "joints":joint_name,
                "frame":{}
                }     

        for frame_index, frame_data in enumerate(GPF):  
                # x, y, zのインデックスを計算
            idx = i * 3
            x = frame_data[idx]
            y = frame_data[idx + 1]
            z = frame_data[idx + 2]
            # ジョイントの座標を追加            
        
            frame_dict["frame"][frame_index] = {"X": x, "Y":y, "Z": z}
        # 構造化データにフレームを追加
        structured_data1.append(frame_dict)


#cProfile.run('animation_instance.animate(0)')
with open('structured_data.json', 'w', encoding='utf-8') as f:
    json.dump(structured_data, f, ensure_ascii=False, indent=4)

with open('structured_data1.json', 'w', encoding='utf-8') as f:
    json.dump(structured_data1, f, ensure_ascii=False, indent=1)



   
for i in range(75,88):   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    frame_to_save = i  # 例として50フレーム目を指定

    # animate関数を直接呼び出し
    animation_instance.animate(frame_to_save)


    # 画像を保存
    plt.savefig(f"frame_{frame_to_save}.png")  # ファイル名にフレーム番号を含める

    # plt.show() は不要 (savefigで保存するので)
    #plt.show()

    # 保存したいフレーム番号を指定


