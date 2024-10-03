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
from typing import Final, Dict, List
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

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
        yaw, _, _ = euler_angles
        to_return = np.concatenate([[cv_data_list[0]], translation_vector, np.array([yaw])])
    
    elif mode == "object":
        ingredient_1 = cmr_coordinate_to_world()
        ingredient_2 = constructCameraPoseGT(cv_data_list, mode=mode)
        homogeneous_matrix = ingredient_1 @ ingredient_2 @ np.linalg.inv(ingredient_1)
        
        translation_vector = homogeneous_matrix[0:3, 3]

        to_return = np.concatenate([cv_data_list[0:2], translation_vector])
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
    if(filtered_df.empty):
        return None
    yaw = float(filtered_df.yaw.iloc[0])
    
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    translation = np.array([filtered_df.x.iloc[0], filtered_df.y.iloc[0]])
    global_x_y = rotation @ np.array([float(local_values[2]), float(local_values[3])]) + translation
    to_ret = local_values
    to_ret[2] = global_x_y[0]
    to_ret[3] = global_x_y[1]
    to_ret.insert(2, category_dict[local_values[1]])
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

from scipy.ndimage import gaussian_filter1d

# Smooth the data x, y, yaw
def smooth_trajectory(df, window_size=5, sigma=2):
    # Initialize an empty list to hold smoothed DataFrames
    smoothed_data = []

    # Group by object_ID
    for object_id, group in df.groupby('object_ID'):
        # Prepare arrays to store smoothed results
        x_smooth = np.zeros(len(group))
        y_smooth = np.zeros(len(group))

        # Loop over the data with a sliding window
        for i in range(len(group)):
            # Define the window range
            start = max(0, i - window_size // 2)
            end = min(len(group), start + window_size)

            # If the window is too small, continue to the next index
            if end - start < 2:
                x_smooth[i] = group['x'].iloc[i]
                y_smooth[i] = group['y'].iloc[i]
                continue
            
            # Apply Gaussian filter to x and y within the window
            x_smooth[start:end] = gaussian_filter1d(group['x'].iloc[start:end], sigma=sigma)
            y_smooth[start:end] = gaussian_filter1d(group['y'].iloc[start:end], sigma=sigma)

        # Store the smoothed results in the group
        group['x_smooth'] = x_smooth
        group['y_smooth'] = y_smooth

        # Calculate smoothed yaw (heading) from smoothed x and y
        group['yaw_smooth'] = np.arctan2(group['y_smooth'].diff(), group['x_smooth'].diff()).fillna(0)

        # Append the smoothed group to the list
        smoothed_data.append(group)

    # Concatenate all the smoothed groups back into a single DataFrame
    smoothed_df = pd.concat(smoothed_data, ignore_index=True)
    return smoothed_df

    
from ccma import CCMA
# This for CCMA smoothing. Upgraded moving average


# Kalman filter function
def kalman_filter(df):
    # Parameters
    dt = 0.05  # time step
    num_states = 4  # [x, y, vx, vy]
    num_measurements = 2  # [x, y]
    
    # Initialize matrices
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])  # State transition matrix

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])  # Measurement matrix

    Q = np.eye(num_states) * 0.1  # Process noise covariance
    R = np.eye(num_measurements) * 0.5  # Measurement noise covariance

    # Create empty lists to hold smoothed data
    results = []

    # Loop through each object_ID
    for obj_id in df['object_ID'].unique():
        # Filter data for the specific object_ID
        obj_data = df[df['object_ID'] == obj_id].copy()
        
        # Initial state
        x = np.array([[obj_data.iloc[0]['x']],
               [obj_data.iloc[0]['y']],
               [0],  # Initial velocity in x
               [0]])  # Initial velocity in y
        
        P = np.eye(num_states)  # Initial uncertainty

        # Lists to hold smoothed positions and yaw angles for the current object
        smoothed_x = []
        smoothed_y = []
        yaw_angles = []

        for i in range(len(obj_data)):
            z = np.array([[obj_data.iloc[i]['x']],
                           [obj_data.iloc[i]['y']]])  # Measurement
            
            # Prediction step
            x = F @ x
            P = F @ P @ F.T + Q
            
            # Update step
            y = z - (H @ x)  # Measurement residual
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
            x = x + K @ y
            P = (np.eye(num_states) - K @ H) @ P
            
            # Store smoothed positions
            smoothed_x.append(x[0, 0])
            smoothed_y.append(x[1, 0])

            # Calculate yaw (heading)
            if i > 0:
                yaw = np.arctan2(smoothed_y[-1] - smoothed_y[-2], smoothed_x[-1] - smoothed_x[-2])
            else:
                yaw = np.arctan2(obj_data.iloc[1]['y'] - obj_data.iloc[0]['y'], obj_data.iloc[1]['x'] - obj_data.iloc[0]['x'])  # Initial yaw
            yaw_angles.append(yaw)

        # Append results for this object to the results list
        results.append(pd.DataFrame({
            'scene_ID': obj_data['scene_ID'].values,
            'frame_ID': obj_data['frame_ID'].values,
            'object_ID': obj_data['object_ID'].values,
            'category': obj_data['category'].values,
            'x': smoothed_x,
            'y': smoothed_y,
            'z': obj_data['z'].values,
            'yaw': yaw_angles
        }))

    # Concatenate results into a single DataFrame
    smoothed_df = pd.concat(results, ignore_index=True)

    return smoothed_df



import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Plot heading difference
def plot_heading_differences(df, title=""):
    # Base directory to save figures
    base_directory = "/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/raw/0000"

    # Ensure 'yaw' column is present
    if 'yaw' not in df.columns:
        raise ValueError("The DataFrame must contain a 'yaw' column.")
    
    # Create a new DataFrame to hold heading differences
    heading_diff = pd.DataFrame(columns=['frame_ID', 'object_ID', 'heading_diff'])

    # Loop through each object_ID
    for obj_id in df['object_ID'].unique():
        obj_data = df[df['object_ID'] == obj_id].copy()
        
        # Calculate the difference in headings
        obj_data['heading_diff'] = obj_data['yaw'].diff().dropna()  # Calculate difference
        obj_data['heading_diff'] = obj_data['heading_diff'].mod(2 * np.pi)  # Ensure difference is within [0, 2π]
        
        # Adjust to ensure non-negative difference
        obj_data['heading_diff'] = np.where(obj_data['heading_diff'] > np.pi,
                                             2 * np.pi - obj_data['heading_diff'],
                                             obj_data['heading_diff'])
        
        obj_data = obj_data[obj_data['heading_diff'].notna()]
        
        
        # Store relevant columns
        if not obj_data.empty:
            heading_diff = pd.concat([heading_diff, obj_data[['frame_ID', 'object_ID', 'heading_diff']]], ignore_index=True)


    # Plotting
    plt.figure(figsize=(12, 6))
    custom_colors = ["blue", "red", "orange", "black"]  # Color map
    custom_colors2 = ["cornflowerblue", "lightcoral", "moccasin", "dimgray"]

    for i, obj_id in enumerate(heading_diff['object_ID'].unique()):
        obj_plot_data = heading_diff[heading_diff['object_ID'] == obj_id]
        plt.plot(obj_plot_data['frame_ID'], obj_plot_data['heading_diff'], label=f'Object ID: {obj_id}', color=custom_colors[i])
        
        # Calculate and plot the average heading difference
        avg_heading_diff = obj_plot_data['heading_diff'].mean()
        plt.axhline(y=avg_heading_diff, linestyle='--', linewidth=1, label=f'Avg {obj_id} (Heading Diff)', color=custom_colors2[i])

    no_underscores = title.replace("_", " ")
    plt.title(f'Heading Differences {no_underscores} by Frame ID')
    plt.xlabel('Frame ID')
    plt.ylabel('Heading Difference (radians)')
    plt.legend()
    plt.grid()
    plot_file_path = os.path.join(base_directory, f'heading_diffs_{title}.png')
    plt.savefig(plot_file_path)
    plt.close()  # Close the figure to free memory



# Visualize the trajectory
def visualize_trajectory(df, title="", df2=None):
    unique_object_ids = df['object_ID'].unique()

    # Base directory to save figures
    base_directory = "/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/raw/0000"

    # Create directory for the object trajectories
    folder_name = "obj_trajectory"
    folder_path = os.path.join(base_directory, folder_name, title)
    no_underscores = title.replace("_", " ")
    
    # Create the folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if df2 is None:
        for object_id in tqdm(unique_object_ids, "Generating Plots for objects"):
            # Filter the DataFrame for the current object ID
            object_data = df[df['object_ID'] == object_id]

            # Sort by frame_ID
            object_data = object_data.sort_values(by='frame_ID')

            # Extract coordinates and yaw
            x = object_data['x'].values
            y = object_data['y'].values
            yaw = object_data['yaw'].values

            for i in range(len(x)):
                # Initialize the plot for the current frame
                plt.figure(figsize=(10, 10))

                # Plot the trajectory up to the current frame
                plt.plot(x[:i+1], y[:i+1], label=f'Object ID: {object_id}', color='blue')

                # Plot yaw as an arrow to visualize heading for each frame
                for j in range(i + 1):
                    plt.arrow(x[j], y[j], 0.5 * np.cos(yaw[j]), 0.5 * np.sin(yaw[j]),
                            head_width=0.2, head_length=0.2, fc='red', ec='red', alpha=0.5)

                # Customize the plot
                plt.title(f'Vehicle Trajectory and Heading - {no_underscores}')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.axis('equal')
                plt.legend()
                plt.grid()

                # Save the figure for the current frame
                plot_file_path = os.path.join(folder_path, f'plot_object_{object_id}_{object_data.iloc[i]["frame_ID"]}.png')
                plt.savefig(plot_file_path)
                plt.close()  # Close the figure to free memory
    else:
        for object_id in unique_object_ids:
            # Filter the DataFrame for the current object ID
            object_data = df[df['object_ID'] == object_id]
            object_data2 = df2[df2['object_ID'] == object_id]
            
            # Sort by frame_ID
            object_data = object_data.sort_values(by='frame_ID')
            object_data2 = object_data2.sort_values(by='frame_ID')

            # Extract coordinates and yaw
            x = object_data['x'].values
            y = object_data['y'].values
            yaw = object_data['yaw'].values
            
            x2 = object_data2['x'].values
            y2 = object_data2['y'].values
            yaw2 = object_data2['yaw'].values

            for i in range(min(len(x), len(x2))):
                # Initialize the plot for the current frame
                plt.figure(figsize=(10, 10))

                # Plot the trajectory up to the current frame
                plt.plot(x[:i+1], y[:i+1], label=f'Before Smoothing ID: {object_id}', color='blue')
                plt.plot(x2[:i+1], y2[:i+1], label=f'After Smoothing ID: {object_id}', color='red')
                
                # Plot yaw as an arrow to visualize heading for each frame
                for j in range(i + 1):
                    plt.arrow(x[j], y[j], 0.5 * np.cos(yaw[j]), 0.5 * np.sin(yaw[j]),
                            head_width=0.2, head_length=0.2, fc='indigo', ec='indigo', alpha=0.5)
                    plt.arrow(x2[j], y2[j], 0.5 * np.cos(yaw2[j]), 0.5 * np.sin(yaw2[j]),
                            head_width=0.2, head_length=0.2, fc='tomato', ec='tomato', alpha=0.5)
                    
                # Customize the plot
                plt.title(f'Vehicle Trajectory and Heading - {no_underscores}')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.axis('equal')
                plt.legend()
                plt.grid()

                # Save the figure for the current frame
                plot_file_path = os.path.join(folder_path, f'plot_object_{object_id}_{object_data.iloc[i]["frame_ID"]}.png')
                plt.savefig(plot_file_path)
                plt.close()  # Close the figure to free memory
    


def apply_ctrv_model_with_rolling_predictions(df):
    # Parameters
    dt = 0.05  # Time step
    prediction_timesteps = 40  # Number of steps to predict (3 seconds)
    history_steps = 8  # Number of steps to use for calculating velocity and turn rate
    
    base_directory = "/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/raw/0000"
    # Create directory for the object trajectories
    folder_name = "CTRV_prediction"
    folder_path = os.path.join(base_directory, folder_name)
    
    # Create the folder if it does not exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # Filter the data for the specific object_ID
    unique_object_ids = df['object_ID'].unique()
    object_1 = unique_object_ids[0]
    obj_data = df[df['object_ID'] == object_1].copy()

    if len(obj_data) < prediction_timesteps + history_steps:
        raise ValueError("Not enough data points to make predictions.")

    # CTRV Model Prediction Loop
    for start_idx in range(len(obj_data) - prediction_timesteps - history_steps):
        # Use the last history_steps for the current prediction
        last_data = obj_data.iloc[start_idx:start_idx + history_steps]
        
        # Extract positions and headings
        positions = last_data[['x', 'y']].values
        headings = last_data['yaw'].values
        
        # Calculate velocity (v0) for the last history_steps
        velocities = []
        for i in range(1, history_steps):
            distance = np.sqrt((positions[i][0] - positions[i - 1][0])**2 + 
                               (positions[i][1] - positions[i - 1][1])**2)
            velocities.append(distance / dt)
        v0 = np.mean(velocities)  # Average velocity

        # Calculate turn rate (omega0) for the last history_steps
        heading_changes = np.diff(headings)
        omega0 = np.mean(heading_changes) / dt  # Average turn rate
        
        # Extract last known position and heading
        x0 = positions[-1][0]
        y0 = positions[-1][1]
        theta0 = headings[-1]

        # Store predictions for this window
        predictions = []

        # Predict the next 60 timesteps
        for t in range(prediction_timesteps):
            x0 += v0 * np.cos(theta0) * dt
            y0 += v0 * np.sin(theta0) * dt
            theta0 += omega0 * dt
            
            predictions.append((x0, y0))

        # Create a plot for the current prediction
        plt.figure(figsize=(12, 6))
        
        # Plot historical data from the beginning up to the current prediction
        historical_x = obj_data['x'].iloc[:start_idx + history_steps+1].values
        historical_y = obj_data['y'].iloc[:start_idx + history_steps+1].values
        plt.plot(historical_x, historical_y, linestyle='--', label='Historical Trajectory', color='lightgrey', linewidth=1)

        # Ground truth trajectory for the next 60 frames
        if (start_idx + prediction_timesteps + history_steps) > (len(obj_data)-1):  
            ground_truth_x = obj_data['x'].iloc[start_idx + history_steps:(len(obj_data)-1) + history_steps].values
            ground_truth_y = obj_data['y'].iloc[start_idx + history_steps:(len(obj_data)-1) + history_steps].values
        else:
            ground_truth_x = obj_data['x'].iloc[start_idx + history_steps:start_idx + prediction_timesteps + history_steps].values
            ground_truth_y = obj_data['y'].iloc[start_idx + history_steps:start_idx + prediction_timesteps + history_steps].values

        # Plot ground truth trajectory
        plt.plot(ground_truth_x, ground_truth_y, label='Ground Truth Trajectory', color='blue', linewidth=2)

        # Plot predicted trajectory
        pred_df_subset = pd.DataFrame(predictions, columns=['predicted_x', 'predicted_y'])
        plt.plot(pred_df_subset['predicted_x'], pred_df_subset['predicted_y'], label='Predicted Trajectory', color='lightcoral')

        plt.title(f'CTRV Model Prediction for Object ID: {object_1} at Frame {start_idx}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid()
        plt.axis('equal')  # Equal aspect ratio to visualize trajectories properly
        
        # Save the figure for the current frame
        plot_file_path = os.path.join(folder_path, f'plot_object_{object_1}_frame_{start_idx}_p_{prediction_timesteps}_h_{history_steps}.png')
        plt.savefig(plot_file_path)
        plt.close()  # Close the figure to free memory

    
# Make static objects static
def ground_static_objects(df, dynamic_obj_list):
    # Get the unique object IDs not in the provided list
    unique_object_ids = df['object_ID'].unique()
    object_ids_to_replicate = [obj_id for obj_id in unique_object_ids if obj_id not in dynamic_obj_list]
    
    for obj_id in object_ids_to_replicate:
        # Find the first occurrence of the object_ID
        first_appearance = df[df['object_ID'] == obj_id].iloc[0]
        first_x = first_appearance['x']
        first_y = first_appearance['y']
        
        # Replicate the x and y values for all occurrences of that object_ID
        df.loc[df['object_ID'] == obj_id, 'x'] = first_x
        df.loc[df['object_ID'] == obj_id, 'y'] = first_y
    
    return df
    
################################################################################################################
# ------------------------------------------------------------------------------------------------------------ #
################################################################################################################

base_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/raw'
data_folders = os.listdir(base_path) # Directory with all the data
# TODO: Assert is directory
output_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/global_data'
maybe_makedirs(output_path) # output data

# Agents that appear and disappear throughout the sequence - missing frames
faulty_agents: Final[Dict[str, str]] = {}
faulty_agents["0004"] = ['41']

# Dynamic Agents - Consider only agents that move in the scene
DYNAMIC_AGENTS: Final[Dict[str, List[int]]] = {
    "0000": [1, 2],                                                                                                 # Single turn
    "0001": [82, 83, 84, 85, 87, 88, 89],                                                                           # 50% straight - Good Scene
    "0002": [1, 5, 6, 7, 8, 9, 13, 15, 16, 17, 18],                                                                 # ~30% straight - intersection
    "0003": [1, 2, 6],                                                                                              # 100% straght
    "0004": [1, 2, 3, 4, 5, 6, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31, 32, 33, 34, 35, 40, 41], # 90% straight
    "0005": [35, 12, 13, 18, 19, 20, 21, 22, 24, 25, 27, 31, 30, 32],                                               # 100% straght
    "0006": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],                                                        # 90% straight
    "0018": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],                            # 100% straght
    "0020": [128, 127, 1, 6, 13, 134, 132],                                                                         # Highway traffic, only right lane is moving straight + 2 nice smooth turns
}


def process_data_cmr():
    
    column_names_cmr = ['scene_ID', 'frame_ID', 'object_ID', 'category', 'x', 'y', 'z', 'yaw']
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
                    data_entry = [data_entry[i] for i in [0, 3, 2, 1, 4, 5, 6, 7]]
                    data.append(data_entry)
        else:
            raise Exception(f"{cmr_gt_path} does not exist.")

    df = pd.DataFrame(data, columns=column_names_cmr)
    df['category'] = df['category'].replace(['CAR', 'BUS', 'BICYCLE'], 'VEHICLE') # For the purpose of the TRAJECTRON++
    df['x'] = df['x'].astype(float); df['y'] = df['y'].astype(float)
    df['frame_ID'] = df['frame_ID'].astype(int); df['scene_ID'] = df['scene_ID'].astype(int)
    return df


def process_data_obj(df_camera_wrong_yaw, create_plots = False):
    df_ego = df_camera_wrong_yaw
    
    # Change camera yaw to the right one and save
    df_camera_save = df_camera_wrong_yaw.drop(columns=['yaw']).copy()
    df_camera_save = create_yaw(df_camera_save)
    df_camera_save.sort_values(by=['scene_ID', 'frame_ID'], inplace=True)
    csv_file_path = os.path.join(output_path, 'kitti_camera_global.csv')
    df_camera_save.to_csv(csv_file_path, index=False)
    print(f"Camera data successfully saved to: \"{csv_file_path}\"")
    #####
    
    columns_to_extract = [1, 2, 7, 8, 9, 10]
    column_names_obj = ['scene_ID', 'frame_ID', 'object_ID', 'category', 'x', 'y', 'z']
    data = []
    combined_df = pd.DataFrame()
    
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
                lines = file.readlines()  # Read all lines into a list
    
                for line in lines[:-1]:  # Iterate through all but the last line
                    # Split the line into values
                    values = line.split()
                    # Extract the specified columns and add them to the data list
                    extracted_values = [values[i - 1] for i in columns_to_extract]

                    normal_values = cv_to_normal(extracted_values, mode="object")
                    data_entry = local_to_global(folder_name, normal_values, df_ego, category_dict)
                    if data_entry is not None:
                        data.append(data_entry)
                df = pd.DataFrame(data, columns=column_names_obj)
                
                # Remove faulty Agents
                if folder_name in faulty_agents:
                    bad_agent_list = faulty_agents[folder_name]
                    df = df[~df['object_ID'].isin(bad_agent_list)]
                    df.reset_index(drop=True, inplace=True)
                
                df['category'] = df['category'].replace(['CAR', 'BUS', 'BICYCLE'], 'VEHICLE') # For the purpose of the TRAJECTRON++
                df['x'] = df['x'].astype(float); df['y'] = df['y'].astype(float)
                df['frame_ID'] = df['frame_ID'].astype(int); df['object_ID'] = df['object_ID'].astype(int)
                df = create_yaw(df)
                
                # Get only dynamic agents
                if folder_name in DYNAMIC_AGENTS:
                    df = df[df['object_ID'].isin(DYNAMIC_AGENTS[folder_name])]
               
                # Keep the Static agents    
                # if folder_name == "0000":
                #     df = df[~df['object_ID'].isin([3])]
                #     df = ground_static_objects(df, [1,2])
                    
                # Plot heading diffs before smoothing
                if create_plots and folder_name == "0000":
                    visualize_trajectory(df, "before_smoothing")
                    plot_heading_differences(df, "before_smoothing")
                    
                # Smoothing
                # df = smooth_trajectory(df)
                df_smoothed = kalman_filter(df)
                
                # Vis predictions
                if create_plots and folder_name == "0000":
                    apply_ctrv_model_with_rolling_predictions(df_smoothed)
                
                # Plot heading diffs after smoothing
                if create_plots and folder_name == "0000":
                    visualize_trajectory(df_smoothed, "after_smoothing")
                    plot_heading_differences(df_smoothed, "after_smoothing")
                
                # Plot heading diffs before and after smoothing
                if create_plots and folder_name == "0000":
                    visualize_trajectory(df, "before_and_after_smoothing", df_smoothed)
                    
                df_smoothed.sort_values(by=['frame_ID', 'object_ID'], inplace=True)
                # If doesnt work, uncomment this
                # df_smoothed.sort_values(by=['object_ID', 'frame_ID'], inplace=True)
                csv_file_path = os.path.join(base_path, folder_name, (folder_name + '.csv'))
                df_smoothed.to_csv(csv_file_path, index=False)
                combined_df = pd.concat([combined_df, df_smoothed], ignore_index=True)
                data = []
        else:
            raise Exception(f"{obj_gt_path} does not exist.")

    csv_file_path = os.path.join(output_path, 'kitti_object_global.csv')
    #combined_df.sort_values(by=['scene_ID', 'object_ID', 'frame_ID'], inplace=True)
    combined_df.sort_values(by=['scene_ID', 'frame_ID', 'object_ID'], inplace=True)
    combined_df['frame_ID'] = combined_df['frame_ID'].astype(str); combined_df['object_ID'] = combined_df['object_ID'].astype(str)
    combined_df.to_csv(csv_file_path, index=False)
    print(f"Object data successfully saved to: \"{csv_file_path}\"")
    
if __name__ == '__main__':
    camera_df = process_data_cmr()
    process_data_obj(camera_df)
    




