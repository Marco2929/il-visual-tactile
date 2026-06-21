import rerun as rr
import rerun.blueprint as rrb
from rerun.urdf import UrdfTree
import zarr
import numpy as np
import cv2
import os

FOLDER_NAME = os.environ.get("FOLDER_NAME", "test_collection")

def visualize_il_dataset(zarr_path, urdf_path):
    rr.init("IL_Robot_Visualizer", spawn=True)

    # 1. Blueprint (Jetzt mit expliziter 3D-Ansicht für End-Effektor und Pointcloud)
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", name="Digital Twin & Actions"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="zed2i/color", name="RGB Kamera"),
                rrb.TimeSeriesView(origin="ur_artus/joints", name="Gelenk-Kinematik")
            )
        )
    )
    rr.send_blueprint(blueprint)

    print("Lade URDF Meshes in Rerun...")
    rr.log_file_from_path(urdf_path, static=True)

    rr.log(
        "camera",
        rr.Transform3D(
            translation=[0.8, 0.0, 0.5], # Anpassen an echte Position
            mat3x3=[[0, 0, -1], [-1, 0, 0], [0, 1, 0]] 
        ),
        static=True
    )
    
    urdf_tree = UrdfTree.from_file_path(urdf_path)

    dataset = zarr.open(zarr_path, mode='r')
    data_group = dataset['data']

    images = data_group['img']
    pointclouds = data_group['point_cloud']
    raw_joint_states = data_group['joint_state']
    
    timestamps = dataset['meta']['episode_ends']
    num_frames = len(images)

    print(f"Starte Wiedergabe von {num_frames} Frames...")
    for i in range(num_frames):
        t_sec = i * (1.0 / 60.0)
        rr.set_time("hardware_clock", duration=t_sec)

        # --- KAMERA & POINTCLOUD ---
        img_frame = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        rr.log("zed2i/color", rr.Image(img_frame))

        # --- FINGER & URDF MATCHING ---
        current_joints = raw_joint_states[i]
        
        # FIX: Mappe die 12 Zarr-Werte exakt auf die Namen, die wir aus deiner Konsole kennen!
        urdf_angles = {
            # UR10e Arm
            'shoulder_pan_joint': current_joints[0],
            'shoulder_lift_joint': current_joints[1],
            'elbow_joint': current_joints[2],
            'wrist_1_joint': current_joints[3],
            'wrist_2_joint': current_joints[4],
            'wrist_3_joint': current_joints[5],
            
            # Inspire Hand (Knöchel)
            'right_inspire_right_thumb_1_joint': current_joints[6],
            'right_inspire_right_thumb_2_joint': current_joints[7],
            'right_inspire_right_index_1_joint': current_joints[8],
            'right_inspire_right_middle_1_joint': current_joints[9],
            'right_inspire_right_ring_1_joint': current_joints[10],
            'right_inspire_right_little_1_joint': current_joints[11],
        }

        # FIX: Mimic Joints (Fingerspitzen). Wir geben ihnen den gleichen Winkel wie 
        # dem Knöchel, damit sich der Finger krümmt, statt steif zu bleiben.
        urdf_angles['right_inspire_right_thumb_3_joint'] = current_joints[7]
        urdf_angles['right_inspire_right_thumb_4_joint'] = current_joints[7]
        urdf_angles['right_inspire_right_index_2_joint'] = current_joints[8]
        urdf_angles['right_inspire_right_middle_2_joint'] = current_joints[9]
        urdf_angles['right_inspire_right_ring_2_joint'] = current_joints[10]
        urdf_angles['right_inspire_right_little_2_joint'] = current_joints[11]

        # --- FIX 3: Graphen-Normierung auf [-1, 1] ---
        for j_name, j_val in urdf_angles.items():
            # Limitiert den Winkel auf [-pi, pi] und skaliert ihn durch Division mit pi
            norm_val = ((j_val + np.pi) % (2 * np.pi) - np.pi) / np.pi
            rr.log(f"ur_artus/joints/{j_name}", rr.Scalars(norm_val))

        # --- FIX 2: NATIVE ROBOT KINEMATICS (Inklusive Fixed Joints!) ---
        for joint in urdf_tree.joints():
            # Wir berechnen jetzt auch die fixed joints (Winkel 0.0), damit die Hand nicht abfällt!
            if joint.joint_type != "fixed":
                angle = urdf_angles.get(joint.name, 0.0)
            else:
                angle = 0.0
                
            transform = joint.compute_transform(angle)
            rr.log("transforms", transform)

if __name__ == "__main__":
    visualize_il_dataset(
        zarr_path=f"/ros2_ws/data/zarr_datasets/{FOLDER_NAME}.zarr", 
        urdf_path="/ros2_ws/src/moveit_ur_hand_config/urdf/inspire_strong_hand_ur10e_flange_workcell.urdf"
    )