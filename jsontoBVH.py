import json
import math
# Define the hierarchy based on joint names
SKELETON_HIERARCHY = {
"Hip": {
"children": ["RHip", "LHip", "Spine"]
},
"RHip": {
"children": ["RKnee"]
},
"RKnee": {
"children": ["RAnkle"]
},
"RAnkle": {
"children": []
},
"LHip": {
"children": ["LKnee"]
},
"LKnee": {
"children": ["LAnkle"]
},
"LAnkle": {
"children": []
},
"Spine": {
"children": ["Thorax"]
},
"Thorax": {
"children": ["Neck/Nose", "LShoulder", "RShoulder"]
},
"Neck/Nose": {
"children": ["Head"]
},
"Head": {
"children": []
},
"LShoulder": {
"children": ["LElbow"]
},
"LElbow": {
"children": ["LWrist"]
},
"LWrist": {
"children": []
},
"RShoulder": {
"children": ["RElbow"]
},
"RElbow": {
"children": ["RWrist"]
},
"RWrist": {
"children": []
}
}
# Define the channels for each joint
# Root joint has position and rotation, others have rotation
CHANNELS = {
"Hip": ["Xposition", "Yposition", "Zposition", "Zrotation", "Yrotation", "Xrotation"],
"RHip": ["Zrotation", "Yrotation", "Xrotation"],
"RKnee": ["Zrotation", "Yrotation", "Xrotation"],
"RAnkle": ["Zrotation", "Yrotation", "Xrotation"],
"LHip": ["Zrotation", "Yrotation", "Xrotation"],
"LKnee": ["Zrotation", "Yrotation", "Xrotation"],
"LAnkle": ["Zrotation", "Yrotation", "Xrotation"],
"Spine": ["Zrotation", "Yrotation", "Xrotation"],
"Thorax": ["Zrotation", "Yrotation", "Xrotation"],
"Neck/Nose": ["Zrotation", "Yrotation", "Xrotation"],
"Head": ["Zrotation", "Yrotation", "Xrotation"],
"LShoulder": ["Zrotation", "Yrotation", "Xrotation"],
"LElbow": ["Zrotation", "Yrotation", "Xrotation"],
"LWrist": ["Zrotation", "Yrotation", "Xrotation"],
"RShoulder": ["Zrotation", "Yrotation", "Xrotation"],
"RElbow": ["Zrotation", "Yrotation", "Xrotation"],
"RWrist": ["Zrotation", "Yrotation", "Xrotation"]
}
def calculate_offset(child, parent, initial_positions):
    """Calculate the offset of child joint relative to parent joint."""
    child_pos = initial_positions[child]
    parent_pos = initial_positions[parent]
    return (
    child_pos['x'] - parent_pos['x'],
    child_pos['y'] - parent_pos['y'],
    child_pos['z'] - parent_pos['z']
)
def write_hierarchy(joint, skeleton_hierarchy, initial_positions, parent=None, bvh_file=None, indent_level=0):
    """Recursively write the hierarchy of joints to the BVH file."""
    indent = " " * indent_level
    if parent is None:
        bvh_file.write(f"HIERARCHY\n")
    if parent is None:
        bvh_file.write(f"{indent}ROOT {joint}\n")
    elif not skeleton_hierarchy[parent]["children"]:
        bvh_file.write(f"{indent}End Site\n{indent}{{\n")
        offset = calculate_offset(joint, parent, initial_positions)
        bvh_file.write(f"{indent} OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
        bvh_file.write(f"{indent}}}\n")
        return
    else:
        bvh_file.write(f"{indent}JOINT {joint}\n")
    bvh_file.write(f"{indent}{{\n")
    offset = calculate_offset(joint, parent, initial_positions) if parent else (0.0, 0.0, 0.0)
    bvh_file.write(f"{indent} OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}\n")
    channels = CHANNELS[joint]
    bvh_file.write(f"{indent} CHANNELS {len(channels)} " + " ".join(channels) + "\n")
    indent_level += 1
    for child in skeleton_hierarchy[joint]["children"]:
        write_hierarchy(child, skeleton_hierarchy, initial_positions, joint, bvh_file)
    indent_level -= 1
    bvh_file.write(f"{indent}}}\n")
    # Initialize indent level as a static variable
    indent_level = 0

def main():
    # Load JSON data
    with open("3d_keypoints.json", "r") as f:
        data = json.load(f)
    frames = data["frames"]
    num_frames = len(frames)
    frame_time = 1.0 / 30.0 # Assuming 30 FPS, adjust as needed
    # Get initial positions from the first frame
    initial_positions = {coord["joint_name"]: coord for coord in frames[0]["coordinates"]}
    # Open BVH file for writing
    with open("output.bvh", "w") as bvh_file:
    # Write Hierarchy
        root_joint = "Hip"
        write_hierarchy(root_joint, SKELETON_HIERARCHY, initial_positions, None, bvh_file)
        # Write Motion
        bvh_file.write("MOTION\n")
        bvh_file.write(f"Frames: {num_frames}\n")
        bvh_file.write(f"Frame Time: {frame_time:.6f}\n")
        # Prepare joint order for motion data
        joint_order = []
        def get_joint_order(joint):
            joint_order.append(joint)
            for child in SKELETON_HIERARCHY[joint]["children"]:
                get_joint_order(child)
        get_joint_order(root_joint)
        # Map joints to their channels
        channels_order = []
        for joint in joint_order:
            channels_order.extend(CHANNELS[joint])
        # Write motion frames
        for frame in frames:
            frame_data = []
            joint_dict = {coord["joint_name"]: coord for coord in frame["coordinates"]}
            for joint in joint_order:
                if joint == "Hip":
                # Root joint positions
                    frame_data.append(joint_dict[joint]["x"])
                    frame_data.append(joint_dict[joint]["y"])
                    frame_data.append(joint_dict[joint]["z"])
                # Here we need to calculate rotations.
                # However, the JSON provides positions, not rotations.
                # For simplicity, we'll set rotations to zero.
                rotation = [0.0, 0.0, 0.0]
                frame_data.extend(rotation)
            # Convert to string with space separation
            frame_line = " ".join([f"{x:.6f}" for x in frame_data])
            bvh_file.write(f"{frame_line}\n")
if __name__ == "__main__":
    main()