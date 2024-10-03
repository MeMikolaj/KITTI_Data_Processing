"""
Process KITTI Camera and Object poses into global coordinates. Save them in the csv files.

The output csv file is in a format:
scene_ID, category, frame_ID, object_ID, x, y, z, roll, pitch, yaw
"""

import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def constructCameraPoseGT(data):
    """
    Args:
        data (pd dataframe 1 row): frame_id tx ty tz qx qy qz qw
    Returns:
        list: SE(3) homogeneous matrix 4x4
    """
    
    # Extract the translation vector
    t = np.array([
        data.tx,
        data.ty,
        data.tz
    ], dtype=np.float64)

    # Construct the rotation matrix
    quaternion = [data.qx, data.qy, data.qz, data.qw]
    r = R.from_quat(quaternion).as_matrix().astype(np.float64)

    Pose = np.eye(4, dtype=np.float64)
    Pose[0:3, 0:3] = r
    Pose[0:3, 3] = t

    return Pose

def cmr_coordinate_to_world() -> np.ndarray:
    """
    Returns:
        np.ndarray: 4x4 matrix for the transformation
    """
    translation_vector = np.array([0.0, 0.0, 0.0])
    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 0:3] = np.array([[0, 0, 1],[-1, 0, 0],[0, -1, 0]])
    transformation_matrix[0:3, 3] = translation_vector
    return(transformation_matrix)

def cv_to_normal(data, mode, ret_rotation=False):
    """ Take a list in cv format and return a normal list in: frameID t1 t2 t3 R11 R12 R13 R21 R22 R23 R31 R32 R33

    Args:
        cv_data_list (np.array): list in cv format: frameID t1 t2 t3 roll pitch yaw # for camera
        cv_data_list (np.array): list in cv format: frameID objectID t1 t2 t3 r1    # for object
        mode (string): "camera" or "object" meaning for ego camera or for object
    """

    ingredient_1 = cmr_coordinate_to_world()
    ingredient_2 = constructCameraPoseGT(data)
    
    homogeneous_matrix = ingredient_1 @ ingredient_2 @ np.linalg.inv(ingredient_1)
    
    translation_vector = homogeneous_matrix[0:3, 3]

    if mode == "camera":
        to_return = np.concatenate([[int(data.frame_id)], translation_vector])
    elif mode == "object" and ret_rotation:
        rotation_matrix = homogeneous_matrix[0:3, 0:3]
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('zyx', degrees=False)  # Yaw, Pitch, Roll
        yaw, pitch, roll = euler_angles
        to_return = np.concatenate([[data.frame_id, data.object_id], translation_vector, np.array([roll, pitch, yaw])])
    elif mode == "object":
        to_return = np.concatenate([[data.frame_id, data.object_id], translation_vector])
    else:
        raise Exception("mode must be \"camera\" if processing for ego camera or \"object\" if processing the objects' data.")
        
    return to_return.tolist()
        
def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise


def local_to_global(folder_name, local_values, df_ego, category_dict):
    index = int(local_values[0])
    filtered_df = df_ego[df_ego['scene_ID'] == int(folder_name)]
    filtered_df = filtered_df[filtered_df['frame_ID'] == index]
    yaw = filtered_df.yaw.iloc[0]
    
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    translation = np.array([filtered_df.x.iloc[0], filtered_df.y.iloc[0]])
    global_x_y = rotation @ np.array([float(local_values[2]), float(local_values[3])]) + translation
    to_ret = local_values
    to_ret[2] = global_x_y[0]
    to_ret[3] = global_x_y[1]
    to_ret[5] = (yaw + float(to_ret[5]) + np.pi) % (2 * np.pi) - np.pi
    to_ret.insert(2, category_dict[str(int(local_values[1]))])
    return [folder_name] + to_ret

def create_yaw(df):
    # Sort the DataFrame by 'scene_ID' and 'frame_ID'
    df.sort_values(by=['scene_ID', 'frame_ID'], inplace=True)
    
    # Initialize the 'yaw' column with NaN
    df['yaw'] = np.nan

    # Calculate 'yaw' for each unique scene_ID and object_ID
    for scene_id in df['scene_ID'].unique():
        scene_df = df[df['scene_ID'] == scene_id]
        
        for obj_id in scene_df['object_ID'].unique():
            obj_df = scene_df[scene_df['object_ID'] == obj_id]
            
            for i in range(len(obj_df) - 1):
                current_index = obj_df.index[i]
                next_index = obj_df.index[i + 1]
                
                # Calculate yaw using atan2
                yaw_value = np.arctan2(
                    df['y'].iloc[next_index] - df['y'].iloc[current_index], 
                    df['x'].iloc[next_index] - df['x'].iloc[current_index] 
                )
                
                # Assign the yaw value to the current index
                df.loc[current_index, 'yaw'] = yaw_value

    # Drop the last values for each object where 'yaw' is NaN
    df.dropna(subset=['yaw'], inplace=True)

    # Reset the index after dropping rows
    df.reset_index(drop=True, inplace=True)

    return df

# ------------------------------------------------------------------------------------------------------------ #

base_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_kitti'
data_folders = os.listdir(base_path) # Directory with all the data
# TODO: Assert is directory
output_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_processed'
maybe_makedirs(output_path) # output data


def process_data():
    
    camera_pose_path   = os.path.join(base_path, 'rgbd_motion_world_backend_camera_pose_log.csv')
    df_cmr_pose = pd.read_csv(camera_pose_path)
    
    obj_pose_path   = os.path.join(base_path, 'rgbd_motion_world_backend_object_pose_log.csv')
    df_obj_pose = pd.read_csv(obj_pose_path)
    
    obj_motion_path = os.path.join(base_path, 'rgbd_motion_world_backend_object_motion_log.csv')
    df_obj_motion = pd.read_csv(obj_motion_path)
    
    
    ####################### CAMERA POSE CV TO NORMAL TO CSV FILE #######################
    
    column_names_cmr = ['scene_ID', 'frame_ID', 'object_ID', 'category', 'x', 'y', 'z']
    data_camera = []
    
    # Change camera data to global in normal x, y, z coordinates.
    for row in df_cmr_pose.itertuples(index=True):
        normal_values = cv_to_normal(row, mode="camera")
        data_entry = [0, 'CAR', 'ego'] + normal_values
        data_entry = [data_entry[i] for i in [0, 3, 2, 1, 4, 5, 6]]
        data_camera.append(data_entry)

    df_cmr = pd.DataFrame(data_camera, columns=column_names_cmr)
    df_cmr['category'] = df_cmr['category'].replace(['CAR', 'BUS', 'BICYCLE'], 'VEHICLE') # For the purpose of the TRAJECTRON++
    df_cmr['x'] = df_cmr['x'].astype(float); df_cmr['y'] = df_cmr['y'].astype(float)
    df_cmr['frame_ID'] = df_cmr['frame_ID'].astype(int); df_cmr['scene_ID'] = df_cmr['scene_ID'].astype(int)
    df_camera_save = create_yaw(df_cmr)
    df_camera_save.sort_values(by=['scene_ID', 'frame_ID'], inplace=True)
    csv_file_path = os.path.join(output_path, 'jesse_camera_global.csv')
    df_camera_save.to_csv(csv_file_path, index=False)
    print(f"Camera data successfully saved to: \"{csv_file_path}\"")
    
    
    ####################### OBJECT POSE CV TO NORMAL #######################
    
    column_names_obj = ['scene_ID', 'frame_ID', 'object_ID', 'category', 'x', 'y', 'z']
    data_obj = []
    
    # Get Objects Categories
    obj_category_path = os.path.join('/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/raw/0000', 'object_category.txt')
    category_dict = {}
    with open(obj_category_path, 'r') as file:
        for line in file:
            # Split the line into values
            values = line.split()
            category_dict[values[0]] = values[1]

    # Change objects data to global in normal x, y, z coordinates.
    for row in df_obj_pose.itertuples(index=True):
        normal_values = cv_to_normal(row, mode="object")
        data_entry = [0] + normal_values
        data_entry.insert(3, category_dict[str(int(normal_values[1]))])
        # data_entry = local_to_global(0, normal_values, df_cmr_pose, category_dict)
        data_obj.append(data_entry)

    df_obj = pd.DataFrame(data_obj, columns=column_names_obj)
    
    df_obj['category'] = df_obj['category'].replace(['CAR', 'BUS', 'BICYCLE'], 'VEHICLE') # For the purpose of the TRAJECTRON++
    df_obj['x'] = df_obj['x'].astype(float); df_obj['y'] = df_obj['y'].astype(float)
    df_obj['frame_ID'] = df_obj['frame_ID'].astype(int); df_obj['object_ID'] = df_obj['object_ID'].astype(int)
    df_obj = create_yaw(df_obj)
    df_obj = df_obj[df_obj['frame_ID'] <= 85] # DATA CONSTRAINTS
    csv_file_path = os.path.join(output_path, 'jesse_object_global.csv')
    df_obj.to_csv(csv_file_path, index=False)
    print(f"Object data successfully saved to: \"{csv_file_path}\"")
    df_obj = df_obj.drop(columns=['yaw']).copy()

    ####################### MOTION POSE CV TO NORMAL #######################
    
    column_names_obj = ['scene_ID', 'frame_ID', 'object_ID', 'category', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']
    data_motion = []

    # Change objects data to global in normal x, y, z coordinates.
    for row in df_obj_motion.itertuples(index=True):
        normal_values = cv_to_normal(row, mode="object", ret_rotation=True)
        data_entry = [0] + normal_values
        data_entry.insert(3, category_dict[str(int(normal_values[1]))])
        # data_entry = local_to_global(0, normal_values, df_cmr_pose, category_dict)
        data_motion.append(data_entry)
    
    df_motion_pose = pd.DataFrame(data_motion, columns=column_names_obj)
    df_motion_pose['frame_ID'] = df_motion_pose['frame_ID'].astype(int)
    df_motion_pose['object_ID'] = df_motion_pose['object_ID'].astype(int)
    df_motion_pose = df_motion_pose[df_motion_pose['frame_ID'] <= 85] # DATA CONSTRAINTS
    # csv_file_path = os.path.join(output_path, 'jesse_motion_global.csv')
    # df_motion_pose.to_csv(csv_file_path, index=False)
    # print(f"Motion data successfully saved to: \"{csv_file_path}\"")
    
    
    ####################### GET VELOCITY #######################
    
    def get_vel(data_obj, data_motion, dt=0.05):
        r = R.from_euler('xyz', [data_motion.roll.iloc[0], data_motion.pitch.iloc[0], data_motion.yaw.iloc[0]]).as_matrix()
        velocities = np.array([data_motion.x.iloc[0], data_motion.y.iloc[0], data_motion.z.iloc[0]]) - (np.eye(3) - r) @ np.array([data_obj.x, data_obj.y, data_obj.z])
        return velocities[0]/dt, velocities[1]/dt, velocities[2]/dt
    
    column_names_vel = ['scene_ID', 'frame_ID', 'object_ID', 'category', 'x', 'y', 'z', 'vx', 'vy']
    data_vels = []
    
    for row in df_obj.itertuples(index=False):
        if row.frame_ID < 85:
            filtered_df = df_motion_pose[(df_motion_pose['frame_ID'] == (row.frame_ID + 1)) & 
                                         (df_motion_pose['object_ID'] == row.object_ID)]
            vx, vy, _ = get_vel(row, filtered_df)
            row_dict = row._asdict()
            row_as_list = list(row_dict.values())
            data_vels.append(row_as_list + [vx, vy])
            
    df_vels = pd.DataFrame(data_vels, columns=column_names_vel)
    
    
    
    ####################### GET ACCELERATION #######################
    
    def get_acc(data_1, data_2, dt=0.05):
        ax = (data_2.vx.iloc[0] - data_1.vx)/dt
        ay = (data_2.vy.iloc[0] - data_1.vy)/dt
        return ax, ay
    
    column_names_acc = ['scene_ID', 'frame_ID', 'object_ID', 'category', 'x', 'y', 'z', 'vx', 'vy', 'ax', 'ay']
    data_acc = []
    for row in df_vels.itertuples(index=False):
        if row.frame_ID < df_vels['frame_ID'].max():
            next_df = df_vels[(df_vels['frame_ID'] == (row.frame_ID + 1)) & 
                              (df_vels['object_ID'] == row.object_ID)]
            if next_df is not None:
                ax, ay = get_acc(row, next_df)
                row_dict = row._asdict()
                row_as_list = list(row_dict.values())
                data_acc.append(row_as_list + [ax, ay])
    
    df_acc = pd.DataFrame(data_acc, columns=column_names_acc)
    # df_acc = df_acc.drop(columns=['roll', 'pitch', 'yaw']).copy()
    
    ####################### SAVE TO CSV FILE #######################
    df_acc['category'] = df_acc['category'].replace(['CAR', 'BUS', 'BICYCLE'], 'VEHICLE') # For the purpose of the TRAJECTRON++
    df_acc['x'] = df_acc['x'].astype(float); df_acc['y'] = df_acc['y'].astype(float)
    df_acc['frame_ID'] = df_acc['frame_ID'].astype(int); df_acc['scene_ID'] = df_acc['scene_ID'].astype(int)
    df_acc = create_yaw(df_acc)
    new_order = ['scene_ID', 'frame_ID', 'object_ID', 'category', 'x', 'y', 'z', 'yaw', 'vx', 'vy', 'ax', 'ay']
    df_acc = df_acc[new_order].copy()
    df_acc.sort_values(by=['scene_ID', 'frame_ID', 'object_ID'], inplace=True)
    
    ####################### SAVE TO CSV FILE #######################
    csv_file_path = os.path.join(output_path, 'jesse_object_final.csv')
    df_acc.to_csv(csv_file_path, index=False)
    print(f"Final (vel and acc) data successfully saved to: \"{csv_file_path}\"")
        
if __name__ == '__main__':
    process_data()
    




