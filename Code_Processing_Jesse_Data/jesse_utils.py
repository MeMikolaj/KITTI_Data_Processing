import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


##############################################################################################################################
##################################         Jesse CV pose to normal (XYZ convention)         ##################################
##############################################################################################################################

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
    """ Take a list in cv format and return a normal list in: frameid t1 t2 t3 R11 R12 R13 R21 R22 R23 R31 R32 R33

    Args:
        cv_data_list (np.array): list in cv format: frameid t1 t2 t3 roll pitch yaw # for camera
        cv_data_list (np.array): list in cv format: frameid objectid t1 t2 t3 r1    # for object
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

def local_to_global(folder_name, local_values, df_ego, category_dict):
    index = int(local_values[0])
    filtered_df = df_ego[df_ego['scene_id'] == int(folder_name)]
    filtered_df = filtered_df[filtered_df['frame_id'] == index]
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

def create_heading(df, drop_last = True):
    # Sort the DataFrame by 'scene_id' and 'frame_id'
    df.sort_values(by=['scene_id', 'frame_id'], inplace=True)
    
    # Initialize the 'heading' column with NaN
    df['heading'] = np.nan

    # Calculate 'heading' for each unique scene_id and object_id
    for scene_id in df['scene_id'].unique():
        scene_df = df[df['scene_id'] == scene_id]
        
        for obj_id in scene_df['object_id'].unique():
            obj_df = scene_df[scene_df['object_id'] == obj_id]
            
            for i in range(len(obj_df) - 1):
                current_index = obj_df.index[i]
                next_index = obj_df.index[i + 1]
                
                # Calculate heading using atan2
                heading_value = np.arctan2(
                    df['y'].iloc[next_index] - df['y'].iloc[current_index], 
                    df['x'].iloc[next_index] - df['x'].iloc[current_index] 
                )
                
                # Assign the heading value to the current index
                df.loc[current_index, 'heading'] = heading_value

    # Drop the last values for each object where 'heading' is NaN
    if drop_last:
        df.dropna(subset=['heading'], inplace=True)

    # Reset the index after dropping rows
    df.reset_index(drop=True, inplace=True)

    return df


#############################################################################################################################
#############################################       Direct DataFrame Utils       ############################################
#############################################################################################################################

# Camera df in CV format to XYZ covention
def camera_to_normal_3D(df, folder_name):
    column_names_cmr = ['scene_id', 'frame_id', 'object_id', 'category', 'x', 'y', 'z']
    data_camera = []
    
    for row in df.itertuples(index=True):
        normal_values = cv_to_normal(row, mode="camera")
        data_entry = [folder_name, 'CAR', 'ego'] + normal_values
        data_entry = [data_entry[i] for i in [0, 3, 2, 1, 4, 5, 6]]
        data_camera.append(data_entry)
    
    df_cmr = pd.DataFrame(data_camera, columns=column_names_cmr)
    return df_cmr


# Object df in CV format to XYZ covention
def object_to_normal_3D(df, category_dict, folder_name):
    column_names_obj = ['scene_id', 'frame_id', 'object_id', 'category', 'x', 'y', 'z']
    data_obj = []

    for row in df.itertuples(index=True):
        normal_values = cv_to_normal(row, mode="object")
        data_entry = [folder_name] + normal_values
        data_entry.insert(3, category_dict[str(int(normal_values[1]))])
        data_obj.append(data_entry)
        
    df_obj = pd.DataFrame(data_obj, columns=column_names_obj)
    return df_obj


# Motion df in CV format to XYZ covention
def motion_to_normal_3D(df, category_dict, dataset_name):
    column_names_obj = ['scene_id', 'frame_id', 'object_id', 'category', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']
    data_motion = []

    # Change objects data to global in normal x, y, z coordinates.
    for row in df.itertuples(index=True):
        normal_values = cv_to_normal(row, mode="object", ret_rotation=True)
        data_entry = [dataset_name] + normal_values
        data_entry.insert(3, category_dict[str(int(normal_values[1]))])
        data_motion.append(data_entry)
    
    df_motion_pose = pd.DataFrame(data_motion, columns=column_names_obj)
    return df_motion_pose


# Change the category name to vehicle for the purpose of trajectron
def categ_to_vehicle(df, cat_list=['CAR', 'BUS', 'BICYCLE']):
    df['category'] = df['category'].replace(cat_list, 'VEHICLE')
    return df


# Set df columns types to correct ones
def set_df_types(df, object_id=False): 
    # Dictionary mapping columns to their desired data types
    column_types = {
        'x': float,
        'y': float,
        'z': float,
        
        'roll': float,
        'pitch': float,
        'yaw': float,
        
        'heading': float,
        
        'scene_id': int,
        'frame_id': int,
        'category': str,
    }
    
    # Iterate through the DataFrame's columns
    for col in df.columns:
        if col in column_types:
            df[col] = df[col].astype(column_types[col])
    
    # object_id can be "ego" or an integer
    if object_id:
        df['object_id'] = df['object_id'].astype(int)
        
    return df


# Calculate Object's velocity and acceleration from motion
def add_vel_acc(df_object, df_motion):
    
    def get_vel(data_obj, data_motion, dt=0.05):
        r = R.from_euler('xyz', [data_motion.roll.iloc[0], data_motion.pitch.iloc[0], data_motion.yaw.iloc[0]]).as_matrix()
        velocities = np.array([data_motion.x.iloc[0], data_motion.y.iloc[0], data_motion.z.iloc[0]]) - (np.eye(3) - r) @ np.array([data_obj.x, data_obj.y, data_obj.z])
        return velocities[0]/dt, velocities[1]/dt, velocities[2]/dt

    def get_acc(data_1, data_2, dt=0.05):
        ax = (data_2.vx.iloc[0] - data_1.vx)/dt
        ay = (data_2.vy.iloc[0] - data_1.vy)/dt
        return ax, ay

    ### Calculate Velocity ###
    column_names_vel = ['scene_id', 'frame_id', 'object_id', 'category', 'x', 'y', 'z', 'vx', 'vy']
    data_vels = []
    
    for row in df_object.itertuples(index=False):
        filtered_df = df_motion[(df_motion['frame_id'] == (row.frame_id + 1)) & (df_motion['object_id'] == row.object_id)]
        if not filtered_df.empty:
            vx, vy, _ = get_vel(row, filtered_df)
        else:
            vx = vy = np.nan
        row_dict = row._asdict()
        row_as_list = list(row_dict.values())
        data_vels.append(row_as_list + [vx, vy])
            
    df_vels = pd.DataFrame(data_vels, columns=column_names_vel)

    ### Calculate Acceleration ###
    column_names_acc = ['scene_id', 'frame_id', 'object_id', 'category', 'x', 'y', 'z', 'vx', 'vy', 'ax', 'ay']
    data_acc = []
    for row in df_vels.itertuples(index=False):
        if row.frame_id < df_vels['frame_id'].max():
            next_df = df_vels[(df_vels['frame_id'] == (row.frame_id + 1)) & (df_vels['object_id'] == row.object_id)]
            if not next_df.empty:
                ax, ay = get_acc(row, next_df)
            else:
                ax = ay = np.nan
            row_dict = row._asdict()
            row_as_list = list(row_dict.values())
            data_acc.append(row_as_list + [ax, ay])
    
    df_acc = pd.DataFrame(data_acc, columns=column_names_acc)
    
    return df_acc


#############################################################################################################################
###############################################         General Utils         ###############################################
#############################################################################################################################

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