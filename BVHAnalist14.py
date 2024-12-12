import bvh
import numpy as np
import bvhio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import transforms3d as t3d
from pyquaternion import Quaternion
import matplotlib.path as mpath
import matplotlib.patches as mpatches

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

root.printTree()




def is_inside_pentagonal_prism(point, vertices, MINZ, MAXZ):
   

    x, y, z = point

    # 高さの範囲内にあるかチェック
    if not (MINZ<= z <= MAXZ):  # 五角柱の高さを0からheightと仮定
        return False
    
    path = mpath.Path(vertices)

    return(path.contains_point(point))  # True




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
        ax.text(WorldPosition[0], WorldPosition[1], WorldPosition[2], joint.Name, fontsize=5)  # ラベルを追加
                    

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


    print(f"{frame_number}point1: {is_inside_pentagonal_prism(point, vertices, minZ, maxZ )}")


ani = animation.FuncAnimation(fig, animate, frames=113, interval=0.0333) # 

plt.show()



for i in range(75,88):   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 保存したいフレーム番号を指定
    frame_to_save = i  # 例として50フレーム目を指定

    # animate関数を直接呼び出し
    animate(frame_to_save)


    # 画像を保存
    plt.savefig(f"frame_{frame_to_save}.png")  # ファイル名にフレーム番号を含める

    # plt.show() は不要 (savefigで保存するので)
    #plt.show()



