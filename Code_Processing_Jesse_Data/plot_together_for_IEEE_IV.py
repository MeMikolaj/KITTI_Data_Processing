"""
Plot data for IV
"""
import sys

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from jesse_utils import *
from typing import Final, List, Dict
import matplotlib.pyplot as plt


base_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_kitti'

output_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_processed/'


def create_plots():
    
    # Read object motion from a file
    obj_path = os.path.join(base_path, 'rgbd_motion_world_backend_object_motion_log.csv')
    df = pd.read_csv(obj_path)
    df = df[df['object_id'] == '2a']


    ######################## Code for Kalman Filter to remove!!!
    gt_x = df_data['gt_x'].values
    gt_y = df_data['gt_y'].values
    
    # Kalman filter Agent
    velocity = df_data['gt_v'].values

    filter_veh = NonlinearKinematicBicycle(dt=0.05, sMeasurement=1.0)
    P_matrix = None
    for i in range(len(x)):
        if i == 0:  # initalize KF
            # initial P_matrix
            P_matrix = np.identity(4)
        elif i < len(x):
            # assign new est values
            x[i] = x_vec_est_new[0][0]
            y[i] = x_vec_est_new[1][0]
            heading[i] = x_vec_est_new[2][0]
            velocity[i] = x_vec_est_new[3][0]

        if i < len(x) - 1:  # no action on last data
            # filtering
            x_vec_est = np.array([[x[i]],
                                    [y[i]],
                                    [heading[i]],
                                    [velocity[i]]])
            z_new = np.array([[x[i + 1]],
                                [y[i + 1]],
                                [heading[i + 1]],
                                [velocity[i + 1]]])
            x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
                x_vec_est=x_vec_est,
                u_vec=np.array([[0.], [0.]]),
                P_matrix=P_matrix,
                z_new=z_new
            )
            P_matrix = P_matrix_new

    # pl = length between 2 points - euc distance
    # curvature, pl, _ = trajectory_curvature(np.stack((x, y), axis=-1))

    if pl < 1.0:  # vehicle is "not" moving
        x = x[0].repeat(max_timesteps + 1)
        y = y[0].repeat(max_timesteps + 1)
        heading = heading[0].repeat(max_timesteps + 1)

    ######################################################################
    
    # Formatting utils for plots
    startup_plotting()

    plot_eucd_xy_1(df, output_path)
    plot_trajectories_1(df, output_path)
    plot_heading_values_1(df, output_path)
    plot_velocity_values_1(df, output_path)

if __name__ == '__main__':
    create_plots()
    





##################################### Plotting Functions #####################################

# Plot Euclidean Distance Poses
def plot_eucd_xy_1(df, output_path, file_folder=""):

    # Plotting
    plt.figure(figsize=(12, 6))


    object_path = os.path.join(output_path, file_folder)
    maybe_makedirs(object_path)
    
    df_data = df.copy()
    
    x_values = np.arange(len(df_data))

    # Estimated
    df_data['euc_d'] = np.sqrt(df_data['x']**2 + df_data['y']**2)
    #df_data['euc_d_diff'] = df_data['euc_d'].diff()  # Calculate difference
    plt.plot(x_values, df_data['euc_d'], label=f'DynoSAM Estimated Data', color='red', zorder=10)
    
    # Ground Truth
    df_data['gt_euc_d'] = np.sqrt(df_data['gt_x']**2 + df_data['gt_y']**2)
    #df_data['gt_euc_d_diff'] = df_data['gt_euc_d'].diff()  # Calculate difference
    plt.plot(x_values, df_data['gt_euc_d'], label=f'Ground Truth Data', color='green', zorder=5)
    
    # Smoothed Ground Truth
    df_data['sgt_euc_d'] = np.sqrt(df_data['sgt_x']**2 + df_data['sgt_y']**2)
    #df_data['sgt_euc_d_diff'] = df_data['sgt_euc_d'].diff()  # Calculate difference
    plt.plot(x_values, df_data['sgt_euc_d'], label=f'Smoothed Ground Truth Data', color='blue', zorder=8)


    plt.title(f'Euclidean distance from (x, y) states to 0. Sequence 0000, Object 2')
    
    plt.xticks(fontsize=20)  # Change x-axis tick label size
    plt.yticks(fontsize=20)  # Change y-axis tick label size

    plt.xlabel('Consecutive Frames', fontsize=26)
    plt.ylabel('Euclidean Distance (m)', fontsize=26)
    
    plt.legend(loc='upper right', fontsize=10)
    
    plt.legend()
    plt.grid()

    plot_file_path = os.path.join(output_path, "IV_xy_euc_kitti0000_2a.png")
    plt.savefig(plot_file_path, format="png", bbox_inches="tight")
    plot_file_path = os.path.join(output_path, "IV_xy_euc_kitti0000_2a.pdf")
    plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory


# Plot Trajectories
def plot_trajectories_1(df, output_path, file_folder=""):

    # Plotting
    plt.figure(figsize=(12, 6))

    object_path = os.path.join(output_path, file_folder)
    maybe_makedirs(object_path)
    
    df_data = df.copy()
    
    x_values = np.arange(len(df_data))

    #### Get Data ####
    x = df_data['x'].values
    y = df_data['y'].values
    gt_x = df_data['gt_x'].values
    gt_y = df_data['gt_y'].values
    sgt_x = df_data['sgt_x'].values
    sgt_y = df_data['sgt_y'].values

    heading = df_data['heading'].values
    gt_heading = df_data['gt_heading'].values
    sgt_heading = df_data['sgt_heading'].values


    # Estimated
    plt.plot(x, y, label=f'DynoSAM Estimated Data', color='red', zorder=10)
    
    # Ground Truth
    plt.plot(gt_x, gt_y, label=f'Ground Truth Data', color='green', zorder=5)
    
    # Smoothed Ground Truth
    plt.plot(sgt_x, sgt_y, label=f'Smoothed Ground Truth Data', color='blue', zorder=8)


    # Arrows
    for j in range(x_values):  
        plt.arrow(x[j], y[j], 0.5 * np.cos(heading[j]), 0.5 * np.sin(heading[j]),
                head_width=0.15, head_length=0.15, fc='darkorange', ec='darkorange', alpha=0.5, zorder=11)
        plt.arrow(gt_x[j], gt_y[j], 0.5 * np.cos(gt_heading[j]), 0.5 * np.sin(gt_heading[j]),
                head_width=0.15, head_length=0.15, fc='springgreen', ec='springgreen', alpha=0.5, zorder=6)
        plt.arrow(sgt_x[j], sgt_y[j], 0.5 * np.cos(sgt_heading[j]), 0.5 * np.sin(sgt_heading[j]),
                head_width=0.15, head_length=0.15, fc='royalblue', ec='royalblue', alpha=0.5, zorder=9)




    plt.title(f'Trajectory. Sequence 0000, Object 2')
    
    plt.xticks(fontsize=20)  # Change x-axis tick label size
    plt.yticks(fontsize=20)  # Change y-axis tick label size

    plt.xlabel('X (m)', fontsize=26)
    plt.ylabel('Y (m)', fontsize=26)
    
    plt.legend(loc='upper right', fontsize=10)
    
    plt.legend()
    plt.grid()

    plot_file_path = os.path.join(output_path, "IV_trajectory_kitti0000_2a.png")
    plt.savefig(plot_file_path, format="png", bbox_inches="tight")
    plot_file_path = os.path.join(output_path, "IV_trajectory_kitti0000_2a.pdf")
    plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory


# Plot heading values
def plot_heading_values_1(df, output_path, file_folder=""):

    # Plotting
    plt.figure(figsize=(12, 6))
    

    object_path = os.path.join(output_path, file_folder)
    maybe_makedirs(object_path)


    x_values = np.arange(len(df))

    plt.plot(x_values, df['heading'], label=f'DynoSAM Estimated Data', color='red', zorder=10)

    plt.plot(x_values, df['gt_heading'], label=f'Ground Truth Data', color='green', zorder=5)

    plt.plot(x_values, df['sgt_heading'], label=f'Smoothed Ground Truth Data', color='blue', zorder=8)

    plt.title(f'Heading Values. Sequence 0000, Object 2')
    
    plt.xticks(fontsize=20)  # Change x-axis tick label size
    plt.yticks(fontsize=20)  # Change y-axis tick label size

    plt.xlabel('Consecutive Frames', fontsize=26)
    plt.ylabel('Heading Values (radians)', fontsize=26)
    
    plt.legend(loc='upper right', fontsize=10)
    
    plt.legend()
    plt.grid()

    plot_file_path = os.path.join(output_path, "IV_heading_val_kitti0000_2a.png")
    plt.savefig(plot_file_path, format="png", bbox_inches="tight")
    plot_file_path = os.path.join(output_path, "IV_heading_val_kitti0000_2a.pdf")
    plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory


# Plot Velocity Values
def plot_velocity_values_1(df, output_path, file_folder=""):

    # Plotting
    plt.figure(figsize=(12, 6))


    object_path = os.path.join(output_path, file_folder)
    maybe_makedirs(object_path)
    
    df_data = df.copy()
    
    x_values = np.arange(len(df_data))

    v = np.sqrt(df_data['vx']**2 + df_data['vy']**2)
    gt_v = np.sqrt(df_data['gt_vx']**2 + df_data['gt_vy']**2)
    sgt_v = np.sqrt(df_data['sgt_vx']**2 + df_data['sgt_vy']**2)

    # Estimated
    plt.plot(x_values, v, label=f'DynoSAM Estimated Data', color='red', zorder=10)
    
    # Ground Truth
    plt.plot(x_values, gt_v, label=f'Ground Truth Data', color='green', zorder=5)
    
    # Smoothed Ground Truth
    plt.plot(x_values, sgt_v, label=f'Smoothed Ground Truth Data', color='blue', zorder=8)


    plt.title(f'Velocity Values. Sequence 0000, Object 2')
    
    plt.xticks(fontsize=20)  # Change x-axis tick label size
    plt.yticks(fontsize=20)  # Change y-axis tick label size

    plt.xlabel('Consecutive Frames', fontsize=26)
    plt.ylabel('Velocity Values (m/s)', fontsize=26)
    
    plt.legend(loc='upper right', fontsize=10)
    
    plt.legend()
    plt.grid()

    plot_file_path = os.path.join(output_path, "IV_v_val_kitti0000_2a.png")
    plt.savefig(plot_file_path, format="png", bbox_inches="tight")
    plot_file_path = os.path.join(output_path, "IV_v_val_kitti0000_2a.pdf")
    plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory