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


##################################### Plotting Functions #####################################

# Plot Euclidean Distance Poses
def plot_eucd_xy_1(df, output_path, file_folder=""):

    # Plotting
    plt.figure(figsize=(12, 6))
    
    df_data = df.copy()
    
    x_values = np.arange(len(df_data))

    # Estimated
    df_data['euc_d'] = np.sqrt(df_data['x']**2 + df_data['y']**2)
    df_data['euc_d_diff'] = df_data['euc_d'].diff()  # Calculate difference
    plt.plot(x_values, df_data['euc_d_diff'], label=f'DynoSAM Estimated', color='red', zorder=10)
    
    # Ground Truth
    df_data['gt_euc_d'] = np.sqrt(df_data['gt_x']**2 + df_data['gt_y']**2)
    df_data['gt_euc_d_diff'] = df_data['gt_euc_d'].diff()  # Calculate difference
    plt.plot(x_values, df_data['gt_euc_d_diff'], label=f'Ground Truth', color='green', zorder=5)
    
    # Smoothed Ground Truth
    df_data['sgt_euc_d'] = np.sqrt(df_data['sgt_x']**2 + df_data['sgt_y']**2)
    df_data['sgt_euc_d_diff'] = df_data['sgt_euc_d'].diff()  # Calculate difference
    plt.plot(x_values, df_data['sgt_euc_d_diff'], label=f'Smoothed Ground Truth', color='blue', zorder=8)


    plt.title(f'Euclidean distance between two consecutive states. Sequence 0000, Object 2')
    
    plt.xticks(fontsize=20)  # Change x-axis tick label size
    plt.yticks(fontsize=20)  # Change y-axis tick label size

    plt.xlabel('Consecutive Frames', fontsize=26)
    plt.ylabel('Euclidean Distance (m)', fontsize=26)
    
    plt.legend(loc='upper right', fontsize=18)
    
    plt.legend()
    plt.grid()

    plot_file_path = os.path.join(output_path, "xy_euc_kitti0000_2a.png")
    plt.savefig(plot_file_path, format="png", bbox_inches="tight")
    plot_file_path = os.path.join(output_path, "xy_euc_kitti0000_2a.pdf")
    plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory


# Plot Trajectories
def plot_trajectories_1(df, output_path, file_folder=""):

    # Plotting
    plt.figure(figsize=(12, 6))
    
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
    plt.plot(x, y, label=f'DynoSAM Estimated', color='red', zorder=10)
    
    # Ground Truth
    plt.plot(gt_x, gt_y, label=f'Ground Truth', color='green', zorder=5)
    
    # Smoothed Ground Truth
    plt.plot(sgt_x, sgt_y, label=f'Smoothed Ground Truth', color='blue', zorder=8)


    # Arrows
    for j in range(len(df_data)):  
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
    
    plt.legend(loc='upper right', fontsize=18)
    
    plt.legend()
    plt.grid()

    plot_file_path = os.path.join(output_path, "trajectory_kitti0000_2a.png")
    plt.savefig(plot_file_path, format="png", bbox_inches="tight")
    plot_file_path = os.path.join(output_path, "trajectory_kitti0000_2a.pdf")
    plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory


# Plot heading values
def plot_heading_values_1(df, output_path, file_folder=""):

    # Plotting
    plt.figure(figsize=(12, 6))


    x_values = np.arange(len(df))

    plt.plot(x_values, df['heading'], label=f'DynoSAM Estimated', color='red', zorder=10)

    plt.plot(x_values, df['gt_heading'], label=f'Ground Truth', color='green', zorder=5)

    plt.plot(x_values, df['sgt_heading'], label=f'Smoothed Ground Truth', color='blue', zorder=8)

    plt.title(f'Heading Values. Sequence 0000, Object 2')
    
    plt.xticks(fontsize=20)  # Change x-axis tick label size
    plt.yticks(fontsize=20)  # Change y-axis tick label size

    plt.xlabel('Consecutive Frames', fontsize=26)
    plt.ylabel('Heading Values (radians)', fontsize=26)
    
    plt.legend(loc='upper right', fontsize=18)
    
    plt.legend()
    plt.grid()

    plot_file_path = os.path.join(output_path, "heading_val_kitti0000_2a.png")
    plt.savefig(plot_file_path, format="png", bbox_inches="tight")
    plot_file_path = os.path.join(output_path, "heading_val_kitti0000_2a.pdf")
    plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory


# Plot Velocity Values
def plot_velocity_values_1(df, output_path, file_folder=""):

    # Plotting
    plt.figure(figsize=(12, 6))
    
    df_data = df.copy()
    
    x_values = np.arange(len(df_data))

    v = np.sqrt(df_data['vx']**2 + df_data['vy']**2)
    gt_v = np.sqrt(df_data['gt_vx']**2 + df_data['gt_vy']**2)
    sgt_v = np.sqrt(df_data['sgt_vx']**2 + df_data['sgt_vy']**2)

    # Estimated
    plt.plot(x_values, v, label=f'DynoSAM Estimated', color='red', zorder=10)
    
    # Ground Truth
    plt.plot(x_values, gt_v, label=f'Ground Truth', color='green', zorder=5)
    
    # Smoothed Ground Truth
    plt.plot(x_values, sgt_v, label=f'Smoothed Ground Truth', color='blue', zorder=8)


    plt.title(f'Velocity Values. Sequence 0000, Object 2')
    
    plt.xticks(fontsize=20)  # Change x-axis tick label size
    plt.yticks(fontsize=20)  # Change y-axis tick label size

    plt.xlabel('Consecutive Frames', fontsize=26)
    plt.ylabel('Velocity Values (m/s)', fontsize=26)
    
    plt.legend(loc='upper right', fontsize=18)
    
    plt.legend()
    plt.grid()

    plot_file_path = os.path.join(output_path, "v_val_kitti0000_2a.png")
    plt.savefig(plot_file_path, format="png", bbox_inches="tight")
    plot_file_path = os.path.join(output_path, "v_val_kitti0000_2a.pdf")
    plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory
    
##################################### END of Plotting Functions #####################################

output_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/IEEE_IV_plots'


def create_plots():
    
    # Read object motion from a file
    obj_path = os.path.join('/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_processed/0000/data/object_pose_motion.csv')
    df = pd.read_csv(obj_path)
    df = df[df['object_id'] == '2a']
    
    # Formatting utils for plots
    startup_plotting()

    plot_eucd_xy_1(df, output_path)
    plot_trajectories_1(df, output_path)
    plot_heading_values_1(df, output_path)
    plot_velocity_values_1(df, output_path)

if __name__ == '__main__':
    create_plots()
    





