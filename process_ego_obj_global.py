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

def constructCameraPoseGT(cv_data_list, mode):
    """
    Args:
        cv_data_list (list): frameID objectID t1 t2 t3 r1 in cv format for object
        cv_data_list (list): frameID t1 t2 t3 R11 R12 R13 R21 R22 R23 R31 R32 R33 for camera
        mode (string): "camera" or "object" meaning for ego camera or for object
    Returns:
        list: SE(3) homogeneous matrix 4x4
    """
    
    if mode == "camera":
        # frameID t1 t2 t3 R11 R12 R13 R21 R22 R23 R31 R32 R33
        # Where ti are the coefficients of camera translation, and R.. is a rotation matrix
        assert(len(cv_data_list) == 13)
        
        # Extract the translation vector
        t = np.array([
            cv_data_list[1],
            cv_data_list[2],
            cv_data_list[3]
        ], dtype=np.float64)

        # Construct the rotation matrix
        R = np.array([
            cv_data_list[4:7],
            cv_data_list[7:10],
            cv_data_list[10:13]
        ], dtype=np.float64)

        Pose = np.eye(4, dtype=np.float64)
        Pose[0:3, 0:3] = R
        Pose[0:3, 3] = t
    elif mode == "object":
        # frameID objectID t1 t2 t3 r1
        # Where ti are the coefficients of 3D object location **t** in camera coordinates, and r1 is the Rotation around
        # Y-axis in camera coordinates.
        assert(len(cv_data_list) == 6)

        # Extract the translation vector
        t = np.array([
            cv_data_list[2],
            cv_data_list[3],
            cv_data_list[4]
        ], dtype=np.float64);

        # Extract rotation angles and convert to radians
        y = float(cv_data_list[5]) + (np.pi / 2)
        x = 0.0
        z = 0.0

        # Compute the rotation matrix elements
        cy = np.cos(y)
        sy = np.sin(y)
        cx = np.cos(x)
        sx = np.sin(x)
        cz = np.cos(z)
        sz = np.sin(z)

        m00 = cy * cz + sy * sx * sz
        m01 = -cy * sz + sy * sx * cz
        m02 = sy * cx
        m10 = cx * sz
        m11 = cx * cz
        m12 = -sx
        m20 = -sy * cz + cy * sx * sz
        m21 = sy * sz + cy * sx * cz
        m22 = cy * cx

        # Construct the rotation matrix
        R = np.array([
            [m00, m01, m02],
            [m10, m11, m12],
            [m20, m21, m22]
        ], dtype=np.float64)

        Pose = np.eye(4, dtype=np.float64)
        Pose[0:3, 0:3] = R
        Pose[0:3, 3] = t
    else:
        raise Exception("mode must be \"camera\" if processing for ego camera or \"object\" if processing the objects' data.")
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

def cv_to_normal(cv_data_list, mode):
    """ Take a list in cv format and return a normal list in: frameID t1 t2 t3 R11 R12 R13 R21 R22 R23 R31 R32 R33

    Args:
        cv_data_list (np.array): list in cv format: frameID t1 t2 t3 roll pitch yaw # for camera
        cv_data_list (np.array): list in cv format: frameID objectID t1 t2 t3 r1    # for object
        mode (string): "camera" or "object" meaning for ego camera or for object
    """
    if mode == "camera":
        ingredient_1 = cmr_coordinate_to_world()
        ingredient_2 = constructCameraPoseGT(cv_data_list, mode=mode)
        homogeneous_matrix = ingredient_1 @ ingredient_2 @ np.linalg.inv(ingredient_1)
        
        translation_vector = homogeneous_matrix[0:3, 3]
        rotation_matrix = homogeneous_matrix[0:3, 0:3]
        
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('zyx', degrees=False)  # Yaw, Pitch, Roll

        # Extract individual angles
        yaw, pitch, roll = euler_angles
        to_return = np.concatenate([[cv_data_list[0]], translation_vector, np.array([roll, pitch, yaw])])
    elif mode == "object":
        ingredient_1 = cmr_coordinate_to_world()
        ingredient_2 = constructCameraPoseGT(cv_data_list, mode=mode)
        homogeneous_matrix = ingredient_1 @ ingredient_2 @ np.linalg.inv(ingredient_1)
        
        translation_vector = homogeneous_matrix[0:3, 3]
        rotation_matrix = homogeneous_matrix[0:3, 0:3]
        
        rotation = R.from_matrix(rotation_matrix)
        euler_angles = rotation.as_euler('zyx', degrees=False)  # Yaw, Pitch, Roll

        # Extract individual angles
        yaw, _, _ = euler_angles
        to_return = np.concatenate([cv_data_list[0:2], translation_vector, np.array([yaw])])
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
    to_ret.insert(2, category_dict[local_values[1]])
    return [folder_name] + to_ret

# ------------------------------------------------------------------------------------------------------------ #

base_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/raw'
data_folders = os.listdir(base_path) # Directory with all the data
# TODO: Assert is directory
output_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/global_data'
maybe_makedirs(output_path) # output data


def process_data_cmr():
    
    column_names_cmr = ['scene_ID', 'frame_ID', 'object_ID', 'category', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']
    data = []
    data_cv = []
    
    for folder_name in tqdm(data_folders, desc="Processing camera ground-truth poses"):
        cmr_gt_path = os.path.join(base_path, folder_name, 'pose_gt.txt')
        
        if os.path.isfile(cmr_gt_path):
            # Open and read the .txt file
            with open(cmr_gt_path, 'r') as file:
                for line in file:
                    # Split the line into values
                    values = line.split()
                    # Frame_id, translation, rotation matrix
                    data_cv = [values[0], values[4], values[8], values[12], values[1], values[2], values[3], values[5], values[6], values[7], values[9], values[10], values[11]]
                    normal_values = cv_to_normal(data_cv, mode="camera")
                    data_entry = [folder_name, 'CAR', 'ego'] + normal_values
                    data_entry = [data_entry[i] for i in [0, 3, 2, 1, 4, 5, 6, 7, 8, 9]]
                    data.append(data_entry)
        else:
            raise Exception(f"{cmr_gt_path} does not exist.")

    df = pd.DataFrame(data, columns=column_names_cmr)
    
    csv_file_path = os.path.join(output_path, 'kitti_camera_global.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"Camera data successfully saved to: \"{csv_file_path}\"")
    return csv_file_path


def process_data_obj(camera_csv_path):
    df_ego = pd.read_csv(camera_csv_path)
    
    columns_to_extract = [1, 2, 7, 8, 9, 10]
    column_names_obj = ['scene_ID', 'frame_ID', 'object_ID', 'category', 'x', 'y', 'z', 'yaw']
    data = []
    
    for folder_name in tqdm(data_folders, desc="Processing objects ground-truth poses"):
        obj_gt_path = os.path.join(base_path, folder_name, 'object_pose.txt')
        obj_category_path = os.path.join(base_path, folder_name, 'object_category.txt')

        category_dict = {}
        if os.path.isfile(obj_category_path):
            with open(obj_category_path, 'r') as file:
                for line in file:
                    # Split the line into values
                    values = line.split()
                    category_dict[values[0]] = values[1]
        else:
            raise Exception(f"{obj_category_path} does not exist.")
            
        if os.path.isfile(obj_gt_path):
            # Open and read the .txt file
            with open(obj_gt_path, 'r') as file:
                for line in file:
                    # Split the line into values
                    values = line.split()
                    # Extract the specified columns and add them to the data list
                    extracted_values = [values[i - 1] for i in columns_to_extract]

                    normal_values = cv_to_normal(extracted_values, mode="object")
                    data_entry = local_to_global(folder_name, normal_values, df_ego, category_dict)
                    data.append(data_entry)
        else:
            raise Exception(f"{obj_gt_path} does not exist.")

    df = pd.DataFrame(data, columns=column_names_obj)
    
    csv_file_path = os.path.join(output_path, 'kitti_object_global.csv')
    df.to_csv(csv_file_path, index=False)
    print(f"Object data successfully saved to: \"{csv_file_path}\"")
    
if __name__ == '__main__':
    camera_csv_path = process_data_cmr()
    process_data_obj(camera_csv_path)
    




