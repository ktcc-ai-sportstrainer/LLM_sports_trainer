import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os 
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
import json 

sys.path.append(os.getcwd())
from lib.utils import normalize_screen_coordinates, camera_to_world
from MotionAGFormer.model.MotionAGFormer import MotionAGFormer

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)


def get_pose2D(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    
    # Add conf score to the last dim
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    output_dir_2d = os.path.join(output_dir, 'input_2D/')
    os.makedirs(output_dir_2d, exist_ok=True)

    output_npz = os.path.join(output_dir_2d, 'keypoints.npz')
    np.savez_compressed(output_npz, reconstruction=keypoints)


def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    if len(names) == 0:
        print("No images found in pose/ directory, skipping img2video.")
        return

    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    video_name = video_path.split('/')[-1].split('.')[0]
    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size) 

    for name in names:
        frame_img = cv2.imread(name)
        videoWrite.write(frame_img)

    videoWrite.release()


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def resample(n_frames):
    # 243フレームにリサンプリング（不足する場合は同じフレームを増やす等）
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1]
    downsample = None
    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
    return clips, downsample

def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data


@torch.no_grad()
def get_pose3D(video_path, output_dir, output_json_path="3d_result.json"):
    """
    メインの3D姿勢推定。最終的に all_3d_coords を JSON出力する。
    さらに同じ構造を呼び出し元に返すか、標準出力に出しても良い。
    """
    # parse known args for model config
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 16, 3, 128, 512, 3
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
    args.use_tcn, args.graph_only = False, False
    args.n_frames = 243
    args = vars(args)

    model = nn.DataParallel(MotionAGFormer(**args)).cuda()

    # load pretrained
    model_path = sorted(glob.glob(os.path.join('MotionAGFormer/checkpoint', 'motionagformer-b-h36m.pth.tr')))[0]
    pre_dict = torch.load(model_path)
    model.load_state_dict(pre_dict['model'], strict=True)

    model.eval()

    # 2D keypoints 
    keypoints_file = os.path.join(output_dir, 'input_2D', 'keypoints.npz')
    keypoints = np.load(keypoints_file, allow_pickle=True)['reconstruction']

    # Clips
    clips, downsample = turn_into_clips(keypoints)

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('\nGenerating 2D pose image...')

    output_dir_2D = os.path.join(output_dir, 'pose2D')
    os.makedirs(output_dir_2D, exist_ok=True)

    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        if img is None:
            continue

        input_2D = keypoints[0][i]  # single frame
        overlay_img = show2Dpose(input_2D, copy.deepcopy(img))
        cv2.imwrite(os.path.join(output_dir_2D, f"{i:04d}_2D.png"), overlay_img)
    
    print('\nGenerating 3D pose...')

    output_dir_3D = os.path.join(output_dir, 'pose3D')
    os.makedirs(output_dir_3D, exist_ok=True)

    all_3d_coords = []
    idx_offset = 0
    ret, temp_img = cap.read()  # read 1 frame to get shape
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset position
    img_size = temp_img.shape if temp_img is not None else (1080,1920,3)

    for idx, clip in enumerate(clips):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0]) 
        input_2D_aug = flip_data(input_2D)
        
        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()

        output_3D_non_flip = model(input_2D) 
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        # handle re-sample
        if idx == len(clips) - 1 and downsample is not None:
            output_3D = output_3D[:, downsample]

        # place hip(0) to origin
        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()  # shape: (n_frames, 17, 3)

        for j, post_out in enumerate(post_out_all):
            frame_index = idx_offset + j
            # camera_to_world transform
            rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)

            # min=0, then scale
            post_out[:, 2] -= np.min(post_out[:, 2])
            scale_val = np.max(post_out)
            if scale_val > 1e-6:
                post_out /= scale_val

            coords_list = post_out.tolist()

            all_3d_coords.append({
                "frame_index": frame_index,
                "coordinates": coords_list
            })

            # 3D visualize
            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05)
            ax = plt.subplot(gs[0], projection='3d')
            show3Dpose(post_out, ax)
            plt.savefig(os.path.join(output_dir_3D, f"{frame_index:04d}_3D.png"),
                        dpi=200, format='png', bbox_inches='tight')
            plt.close(fig)

        idx_offset += len(post_out_all)
        
    print('Generating 3D pose successful!')

    # Write JSON
    final_json = {
        "video_file": video_path,
        "total_frames": len(all_3d_coords),
        "frames": all_3d_coords
    }
    with open(os.path.join(output_dir, output_json_path), 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=4)

    # さらに標準出力にもJSONを書き出し:
    print(json.dumps(final_json, ensure_ascii=False, indent=4))

    # create demo video
    print('\nGenerating demo...')
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    output_dir_pose = os.path.join(output_dir, 'pose')
    os.makedirs(output_dir_pose, exist_ok=True)

    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        # Crop example
        if image_2d.shape[0] < image_2d.shape[1]:
            edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
            image_2d = image_2d[:, edge:(image_2d.shape[1] - edge)]

        # 3D image crop
        edge = 130
        if image_3d.shape[0] > edge*2 and image_3d.shape[1] > edge*2:
            image_3d = image_3d[edge:image_3d.shape[0] - edge,
                                 edge:image_3d.shape[1] - edge]

        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize=12)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize=12)

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.savefig(os.path.join(output_dir_pose, f"{i:04d}_pose.png"),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='Path to input video')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    parser.add_argument('--out_json', type=str, default='3d_result.json', help='Output JSON file name')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = args.video
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f'./run/output/{video_name}/'
    os.makedirs(output_dir, exist_ok=True)

    # 1) 2D keypoints extraction
    get_pose2D(video_path, output_dir)

    # 2) 3D pose estimation + JSON output
    get_pose3D(video_path, output_dir, output_json_path=args.out_json)

    # 3) 2D/3D combined video
    img2video(video_path, output_dir)

    print('Generating demo successful!')
