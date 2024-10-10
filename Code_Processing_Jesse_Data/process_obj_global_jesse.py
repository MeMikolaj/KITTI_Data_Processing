"""
Process KITTI Camera and Object poses into global coordinates. Save them in the csv files.
Calculate the velcoties and accelerations from motion and add them to the object csv, save it.
"""
import sys

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from jesse_utils import *
from typing import Final, List, Dict


base_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_kitti'
data_folders = os.listdir(base_path)

output_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_processed/'
category_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/raw'


def process_data(plot_estimated_traj=True, plot_gt_traj=True, plot_together_traj=True, 
                 plot_estimated_headings=True, plot_gt_headings=True, plot_together_headings=True,
                 plot_estimated_values=True, plot_gt_values=True, plot_together_values=True,
                 plot_estimated_CTRV=True,):
    """ Take 3 csv files (camera pose, object pose, object motion) and arguments.
        Change data to XYZ convention, generate plots etc

    Args:
        name (type): description
    """
    
    print(f"""
          ''''''''''''''''''''''''''''''''''''''''''''
          - Heading the same as previous one if delta_x < 0.1 and delta_y < 0.1
          - Agents with missing frames treated as many objects (1 -> 1a, 1b, 1c)
          
          PLOT TRAJECTORIES
          - Estimated:                {plot_estimated_traj}
          - Groung-Truth:             {plot_gt_traj}
          - Est and GT together:      {plot_together_traj}
          
          PLOT HEADINGS DIFFERENCES
          - Estimated:                {plot_estimated_headings}
          - Groung-Truth:             {plot_gt_headings}
          - Est and GT together:      {plot_together_headings}
          
          PLOT HEADINGS VALUES
          - Estimated:                {plot_estimated_values}
          - Groung-Truth:             {plot_gt_values}
          - Est and GT together:      {plot_together_values}
          
          PLOT CONSTANT TURN RATE AND VELOCITY
          - Estimated:                {plot_estimated_CTRV}
          ''''''''''''''''''''''''''''''''''''''''''''
          """)
    
    
    
    # For each dataset process the data
    for folder_name in tqdm(data_folders, desc="Processing datasets"):
        
        # Get dataset name
        dataset_name = folder_name.split('_')[1] # 0000 from kitti_0000 or 0006 from kitti_0006
        
        # Create an output folder
        maybe_makedirs(os.path.join(output_path, dataset_name, 'data'))
        
        # Read camera pose from a file
        camera_pose_path   = os.path.join(base_path, folder_name, 'rgbd_motion_world_backend_camera_pose_log.csv')
        df_cmr_pose = pd.read_csv(camera_pose_path)
        
        # Read object pose from a file
        obj_pose_path   = os.path.join(base_path, folder_name, 'rgbd_motion_world_backend_object_pose_log.csv')
        df_obj_pose = pd.read_csv(obj_pose_path)
        
        # Read object motion from a file
        obj_motion_path = os.path.join(base_path, folder_name, 'rgbd_motion_world_backend_object_motion_log.csv')
        df_obj_motion = pd.read_csv(obj_motion_path)
        
        
        ####################### GET OBJECT CATEGORIES #######################
        obj_category_path = os.path.join(category_path + '/' + dataset_name, 'object_category.txt')
        category_dict = {}
        with open(obj_category_path, 'r') as file:
            for line in file:
                # Split the line into values
                values = line.split()
                category_dict[values[0]] = values[1]
                    
                    
        ####################### PROCESS ESTIMATED and GT DATA #######################
            
        ############# CAMERA POSE CV TO NORMAL TO CSV FILE #############

        df_cmr = camera_to_normal_3D(df_cmr_pose, dataset_name)   # CV to Normal
        df_cmr = categ_to_vehicle(df_cmr)                         # Category (bus, car, bike) -> Vehicle
        df_cmr = set_df_types(df_cmr, include_obj_id=False)       # Casting columns to their type
        
        df_cmr = create_heading(df=df_cmr, create_turn_rate=True) # Add heading column
        
        # Save Data
        csv_file_path = os.path.join(output_path, dataset_name, 'data', 'camera_pose.csv')
        df_cmr.dropna(inplace=True)
        df_cmr.to_csv(csv_file_path, index=False)
        
        ############# OBJECT POSE CV TO NORMAL #############

        df_obj = object_to_normal_3D(df_obj_pose, category_dict, dataset_name) # CV to Normal
        df_obj = categ_to_vehicle(df_obj)                                      # Category (bus, car, bike) -> Vehicle
        df_obj = set_df_types(df_obj, include_obj_id=True)                     # Casting columns to their type
        
        df_obj_save = df_obj
        
        # Save Data
        df_obj_save = fix_missing_frames(df_obj_save) 
        df_obj_save = create_heading(df=df_obj_save, create_turn_rate=True)
        df_obj_save.dropna(inplace=True)
        df_obj_save.sort_values(by=['scene_id', 'frame_id', 'object_id'], inplace=True)
        csv_file_path = os.path.join(output_path, dataset_name, 'data', 'object_poses.csv')
        df_obj_save.to_csv(csv_file_path, index=False)
 
        ############# OBJECT MOTION INCLUSIVE #############
        
        # Cv to normal
        df_motion_pose = motion_to_normal_3D(df_obj_motion, category_dict, dataset_name) # CV to Normal
        df_motion_pose = set_df_types(df_motion_pose, include_obj_id=True)               # Casting columns to their type
        
        # Get Velocity and Acceleration from motion
        df_acc = add_vel_acc(df_obj, df_motion_pose)        # Vel and Acc
        
        # Improve the data
        df_acc = categ_to_vehicle(df_acc)                   # Category (bus, car, bike) -> Vehicle
        df_acc = set_df_types(df_acc, include_obj_id=True)  # Casting columns to their type
        
        df_acc.dropna(inplace=True)
        df_acc = fix_missing_frames(df_acc)                       # Update object_id considering missing frames
        df_acc = create_heading(df_acc, create_turn_rate=True)    # Add heading column
        
        # Change the order
        new_order = ['scene_id', 'frame_id', 'object_id', 'category', 'x', 'y', 'z', 'heading', 'vx', 'vy', 'ax', 'ay', 'turn_rate', 'gt_x', 'gt_y', 'gt_z', 'gt_heading', 'gt_vx', 'gt_vy', 'gt_ax', 'gt_ay', 'gt_turn_rate']
        df_acc = df_acc[new_order].copy()
        df_acc.sort_values(by=['scene_id', 'frame_id', 'object_id'], inplace=True)
        
        # Save Data
        csv_file_path = os.path.join(output_path, dataset_name, 'data', 'object_pose_motion.csv')
        df_acc.dropna(inplace=True)
        df_acc.to_csv(csv_file_path, index=False)
        
        
        
        ####################### Plot Trajectories #######################
        if plot_estimated_traj:
            plot_poses(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_camera', plot_estimated=True) # Camera
            plot_poses(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_objects_motion', plot_estimated=True) # Motion
            plot_poses(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_objects_poses', plot_estimated=True) # Object estimated poses
            
        if plot_gt_traj:
            plot_poses(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_camera', plot_gt=True) # Camera
            plot_poses(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_objects_motion', plot_gt=True) # Motion
            plot_poses(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_objects_poses', plot_gt=True) # Object estimated poses
            
        if plot_together_traj:
            plot_poses(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_camera', plot_estimated=True, plot_gt=True) # Camera
            plot_poses(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_objects_motion', plot_estimated=True, plot_gt=True) # Motion
            plot_poses(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_objects_poses', plot_estimated=True, plot_gt=True) # Object estimated poses
            
            
        ####################### Plot Heading Differences #######################
        if plot_estimated_headings:
            plot_heading_differences(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_camera', plot_estimated=True) # Camera
            plot_heading_differences(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_objects_motion', plot_estimated=True) # Motion
            plot_heading_differences(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_objects_poses', plot_estimated=True) # Object estimated poses
            
        if plot_gt_headings:
            plot_heading_differences(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_camera', plot_gt=True) # Camera
            plot_heading_differences(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_objects_motion', plot_gt=True) # Motion
            plot_heading_differences(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_objects_poses', plot_gt=True) # Object estimated poses
            
        if plot_together_headings:
            plot_heading_differences(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_camera', plot_estimated=True, plot_gt=True) # Camera
            plot_heading_differences(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_objects_motion', plot_estimated=True, plot_gt=True) # Motion
            plot_heading_differences(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_objects_poses', plot_estimated=True, plot_gt=True) # Object estimated poses
            
        ####################### Plot Heading Values #######################
        if plot_estimated_values:
            plot_heading_values(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_camera', plot_estimated=True) # Camera
            plot_heading_values(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_objects_motion', plot_estimated=True) # Motion
            plot_heading_values(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_objects_poses', plot_estimated=True) # Object estimated poses
            
        if plot_gt_values:
            plot_heading_values(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_camera', plot_gt=True) # Camera
            plot_heading_values(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_objects_motion', plot_gt=True) # Motion
            plot_heading_values(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_objects_poses', plot_gt=True) # Object estimated poses
            
        if plot_together_values:
            plot_heading_values(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_camera', plot_estimated=True, plot_gt=True) # Camera
            plot_heading_values(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_objects_motion', plot_estimated=True, plot_gt=True) # Motion
            plot_heading_values(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_objects_poses', plot_estimated=True, plot_gt=True) # Object estimated poses
            
        ####################### Plot CTRV #######################
        if plot_estimated_CTRV:
            plot_ctrv_model(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_objects_motion', vis_hist_used=True) # Motion
            
        
if __name__ == '__main__':
    process_data()
    




