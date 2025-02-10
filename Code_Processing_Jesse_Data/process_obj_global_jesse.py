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
from kalman_filter import NonlinearKinematicBicycle


base_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_kitti'
data_folders = os.listdir(base_path)

output_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_processed/'
category_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/raw'


def process_data(plot_estimated_traj=True, plot_gt_traj=False, plot_together_traj=True, 
                 plot_estimated_headings=True, plot_gt_headings=False, plot_together_headings=True,
                 plot_estimated_values=True, plot_gt_values=False, plot_together_values=True,
                 plot_estimated_velocity=True, plot_gt_velocity=False, plot_together_velocity=True,
                 plot_estimated_CTRV=False, plot_tron=False, plot_xy_pose=True):
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
          
          PLOT POSE XY EUC D
          - Plot or not:              {plot_xy_pose}
          
          PLOT HEADINGS DIFFERENCES
          - Estimated:                {plot_estimated_headings}
          - Groung-Truth:             {plot_gt_headings}
          - Est and GT together:      {plot_together_headings}
          
          PLOT HEADINGS VALUES
          - Estimated:                {plot_estimated_values}
          - Groung-Truth:             {plot_gt_values}
          - Est and GT together:      {plot_together_values}
          
          PLOT VELOCITIES
          - Estimated:                {plot_estimated_velocity}
          - Groung-Truth:             {plot_gt_velocity}
          - Est and GT together:      {plot_together_velocity}
          
          PLOT CONSTANT TURN RATE AND VELOCITY
          - Estimated:                {plot_estimated_CTRV}

          PLOT FOR TRON PAPER (heading, (x, y), velocity, acceleration)
          - plot_tron:                {plot_tron}
          
          ''''''''''''''''''''''''''''''''''''''''''''
          """)
    
    
    
    # For each dataset process the data
    for folder_name in tqdm(data_folders, desc="Processing datasets"):
        
        # Get dataset name
        dataset_name = folder_name.split('_')[1] # 0000 from kitti_0000 or 0006 from kitti_0006
        
        if dataset_name != "0018":
            continue
        # Only Process 0000 
        # if dataset_name != "0061" and dataset_name != "0103" and dataset_name != "0655" and  dataset_name != "0757":
        #     continue
                    
        
        # Create an output folder
        maybe_makedirs(os.path.join(output_path, dataset_name, 'data'))
        
        # Read camera pose from a file
        camera_pose_path   = os.path.join(base_path, folder_name, 'rgbd_motion_world_backend_camera_pose_log.csv')
        df_cmr_pose = pd.read_csv(camera_pose_path)
        
        # Read object pose from a file
        obj_pose_path   = os.path.join(base_path, folder_name, 'rgbd_motion_world_backend_object_pose_log.csv')
        df_obj_pose = pd.read_csv(obj_pose_path)
        #df_obj_pose = df_obj_pose[df_obj_pose['object_id'] == 32]
        
        # Read object motion from a file
        obj_motion_path = os.path.join(base_path, folder_name, 'rgbd_motion_world_backend_object_motion_log.csv')
        df_obj_motion = pd.read_csv(obj_motion_path)
        #df_obj_motion = df_obj_motion[df_obj_motion['object_id'] == 32]
        
        # print(dataset_name)
        # print(df_obj_pose['object_id'].unique())
        
        ######### NuScenes Mini #########
        # nusc_path = os.path.join('/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/NuscMini_scene_61_id_c1958768d48640948f6053d04cffd35b.csv')
        # df_nusc = pd.read_csv(nusc_path)
        # df_nusc = df_nusc.rename(columns={'node_id': 'object_id'})
        
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
        
        df_cmr_to_merge = df_cmr.copy()
        if dataset_name in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0018', '0020']:
            df_cmr = create_heading(df=df_cmr, stabilize_heading=True, create_turn_rate=False) # Add heading column
        else:
            df_cmr = create_heading(df=df_cmr, stabilize_heading=False, create_turn_rate=False) # Add heading column
        
        
        # Save Data
        csv_file_path = os.path.join(output_path, dataset_name, 'data', 'camera_pose.csv')
        df_cmr.dropna(inplace=True)
        
        df_cmr.to_csv(csv_file_path, index=False)
        
        ############# OBJECT POSE CV TO NORMAL #############

        df_obj = object_to_normal_3D(df_obj_pose, category_dict, dataset_name) # CV to Normal
        
        
        df_obj = categ_to_vehicle(df_obj)                                      # Category (bus, car, bike) -> Vehicle
        df_obj = set_df_types(df_obj, include_obj_id=True)                     # Casting columns to their type
        
        # Add noise to check how trajectron behaves
        # for index, row in df_obj.iterrows():
        #     noise_x = np.random.normal(0, 0.04)
        #     noise_y = np.random.normal(0, 0.04)
        #     df_obj.at[index, 'x'] += noise_x
        #     df_obj.at[index, 'y'] += noise_y
    
 
        ############# OBJECT MOTION INCLUSIVE #############
        
        # Cv to normal
        df_motion_pose = motion_to_normal_3D(df_obj_motion, category_dict, dataset_name) # CV to Normal
        df_motion_pose = replicate_missing_motion_frames(df_motion_pose) ### !!!!!!!!!!!! ### Replicate missing frames for motion of objects that disappear for a bit
        df_motion_pose = set_df_types(df_motion_pose, include_obj_id=True)               # Casting columns to their type
        
        
        
        ################ Use motion data to change the object data starting from when the object disappears.
        df_obj = recalculate_pose_using_motion(df_obj, df_motion_pose) ### !!!!! ###
        
        # Save Data
        df_obj_save = df_obj
        df_obj_save = fix_missing_frames(df_obj_save)
        if dataset_name in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0018', '0020']:
            df_obj_save = create_heading(df=df_obj_save, stabilize_heading=True, create_turn_rate=False)
        else:
            df_obj_save = create_heading(df=df_obj_save, stabilize_heading=False, create_turn_rate=False)
        
        #df_obj_save.dropna(inplace=True)
        df_obj_save.sort_values(by=['scene_id', 'frame_id', 'object_id'], inplace=True)
        csv_file_path = os.path.join(output_path, dataset_name, 'data', 'object_poses.csv')
        df_obj_save.to_csv(csv_file_path, index=False)
        
        
        # Get Velocity and Acceleration from motion
        df_acc = add_vel_acc(df_obj, df_motion_pose)        # Vel and Acc
        # df_acc['x'] = df_acc['x']*6
        # df_acc['y'] = df_acc['y']*6
        # df_acc['vx'] = df_acc['vx']*6
        # df_acc['vy'] = df_acc['vy']*6
        df_acc.sort_values(by=['scene_id', 'frame_id', 'object_id'], inplace=True)
        
        
        # Improve the data
        df_acc = categ_to_vehicle(df_acc)                   # Category (bus, car, bike) -> Vehicle
        df_acc = set_df_types(df_acc, include_obj_id=True)  # Casting columns to their type

        #df_acc.dropna(inplace=True)
        df_acc = fix_missing_frames(df_acc)                     # Update object_id considering missing frames
        
        if dataset_name in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0018', '0020']:
            df_acc = create_heading(df_acc, stabilize_heading=True, create_turn_rate=False)    # Add heading column
        else:
            df_acc = create_heading(df_acc, stabilize_heading=False, create_turn_rate=False)    # Add heading column
        
        
        
        ################################## Add Camera to data ######################
        df_cmr_to_merge['object_id'] = '999a'
        x = df_cmr_to_merge['x'].values
        y = df_cmr_to_merge['y'].values
        
        df_cmr_to_merge['vx'] = np.append(np.nan, (x[1:] - x[:-1]) / 0.05)
        df_cmr_to_merge['vy'] = np.append(np.nan, (y[1:] - y[:-1]) / 0.05)
        df_cmr_to_merge['v'] = np.sqrt(df_cmr_to_merge['vx']**2 + df_cmr_to_merge['vy']**2)
            
        # df_cmr_to_merge['ax'] = np.append(np.nan, (df_cmr_to_merge['vx'].values[1:] - df_cmr_to_merge['vx'].values[:-1]) / 0.05)
        # df_cmr_to_merge['ay'] = np.append(np.nan, (df_cmr_to_merge['vy'].values[1:] - df_cmr_to_merge['vy'].values[:-1]) / 0.05)
        
        
        gt_x = df_cmr_to_merge['gt_x'].values
        gt_y = df_cmr_to_merge['gt_y'].values
        
        df_cmr_to_merge['gt_vx'] = np.append(np.nan, (gt_x[1:] - gt_x[:-1]) / 0.05)
        df_cmr_to_merge['gt_vy'] = np.append(np.nan, (gt_y[1:] - gt_y[:-1]) / 0.05)
        df_cmr_to_merge['gt_v'] = np.sqrt(df_cmr_to_merge['gt_vx']**2 + df_cmr_to_merge['gt_vy']**2)
            
        df_cmr_to_merge['gt_ax'] = np.append(np.nan, (df_cmr_to_merge['gt_vx'].values[1:] - df_cmr_to_merge['gt_vx'].values[:-1]) / 0.05)
        df_cmr_to_merge['gt_ay'] = np.append(np.nan, (df_cmr_to_merge['gt_vy'].values[1:] - df_cmr_to_merge['gt_vy'].values[:-1]) / 0.05)
        if dataset_name in ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0018', '0020']:
            df_cmr_to_merge = create_heading(df_cmr_to_merge, stabilize_heading=True, create_turn_rate=False)
        else:
            df_cmr_to_merge = create_heading(df_cmr_to_merge, stabilize_heading=False, create_turn_rate=False)
        df_cmr_to_merge.dropna(inplace=True)
        
        ################################## KALMAN ##################################
        df_new_copy = df_cmr_to_merge.copy()
        
        x = df_new_copy['x'].values
        y = df_new_copy['y'].values
        heading = df_new_copy['heading'].values
        velocity = df_new_copy['v'].values
        
        filter_veh = NonlinearKinematicBicycle(dt=0.05, sMeasurement=1.0)
        P_matrix = None
        for i in range(len(gt_x)):
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
        # End of Kalman Filter
        # Start of adding sgt to the dataframe
        
        vx = (x[1:] - x[:-1]) / 0.05
        vy = (y[1:] - y[:-1]) / 0.05
        

        df_cmr_to_merge['x'] = x
        df_cmr_to_merge['y'] = y
        df_cmr_to_merge['heading'] = heading
        
        df_cmr_to_merge['vx'] = np.append(np.nan, vx)
        df_cmr_to_merge['vy'] = np.append(np.nan, vy)
        # df_cmr_to_merge['v'] = np.sqrt(df_cmr_to_merge['vx']**2 + df_cmr_to_merge['vy']**2)
            
        df_cmr_to_merge['ax'] = np.append(np.nan, (df_cmr_to_merge['vx'].values[1:] - df_cmr_to_merge['vx'].values[:-1]) / 0.05)
        df_cmr_to_merge['ay'] = np.append(np.nan, (df_cmr_to_merge['vy'].values[1:] - df_cmr_to_merge['vy'].values[:-1]) / 0.05)
        
        # df_acc = pd.concat([df_acc, df_cmr_to_merge], axis=0, ignore_index=True)
        ############################################################################
        
        
        # Change the order
        new_order = ['scene_id', 'frame_id', 'object_id', 'category', 'x', 'y', 'z', 'heading', 'vx', 'vy', 'ax', 'ay', 'gt_x', 'gt_y', 'gt_z', 'gt_heading', 'gt_vx', 'gt_vy', 'gt_ax', 'gt_ay']
        df_acc = df_acc[new_order].copy()
        df_acc.sort_values(by=['scene_id', 'frame_id', 'object_id'], inplace=True)
        
        ########### Get center of the road values for map
        # df_acc_obj = df_acc[df_acc['object_id'] == '32a']
        # np_array = np.column_stack((df_acc_obj['gt_x'], df_acc_obj['gt_y']))
        # np.set_printoptions(precision=8, suppress=True)
        # print(len(np_array))
        # print("----")
        # formatted_output = ', '.join([f'[{x}, {y}]' for x, y in np_array])
        # print(formatted_output)
        # break
        ###########
        
        ###################################################################### 
        # Code for Kalman Filter
        df_acc['gt_v'] = np.sqrt(df_acc['gt_vx']**2 + df_acc['gt_vy']**2)
        df_acc['v'] = np.sqrt(df_acc['vx']**2 + df_acc['vy']**2)
        df_acc.dropna(inplace=True)
            
        df_to_return = pd.DataFrame(columns=['scene_id', 'frame_id', 'object_id', 'category', 'x', 'y', 'z', 'heading', 'gt_x', 'gt_y', 'gt_z', 'gt_heading', 'vx', 'vy', 'ax', 'ay', 'v', 'gt_vx', 'gt_vy', 'gt_v', 'gt_ax', 'gt_ay', 'sgt_x', 'sgt_y', 'sgt_heading', 'sgt_vx', 'sgt_vy', 'sgt_v', 'sgt_ax', 'sgt_ay'])
        

        for unique_object_id in df_acc['object_id'].unique():
        
            df_new = df_acc[df_acc['object_id'] == unique_object_id].copy()
            df_new_copy = df_new.copy()
            
            gt_x = df_new_copy['gt_x'].values
            gt_y = df_new_copy['gt_y'].values
            gt_heading = df_new_copy['gt_heading'].values
            gt_velocity = df_new_copy['gt_v'].values
            
            filter_veh = NonlinearKinematicBicycle(dt=0.05, sMeasurement=1.0)
            P_matrix = None
            for i in range(len(gt_x)):
                if i == 0:  # initalize KF
                    # initial P_matrix
                    P_matrix = np.identity(4)
                elif i < len(gt_x):
                    # assign new est values
                    gt_x[i] = x_vec_est_new[0][0]
                    gt_y[i] = x_vec_est_new[1][0]
                    gt_heading[i] = x_vec_est_new[2][0]
                    gt_velocity[i] = x_vec_est_new[3][0]

                if i < len(gt_x) - 1:  # no action on last data
                    # filtering
                    x_vec_est = np.array([[gt_x[i]],
                                            [gt_y[i]],
                                            [gt_heading[i]],
                                            [gt_velocity[i]]])
                    z_new = np.array([[gt_x[i + 1]],
                                        [gt_y[i + 1]],
                                        [gt_heading[i + 1]],
                                        [gt_velocity[i + 1]]])
                    x_vec_est_new, P_matrix_new = filter_veh.predict_and_update(
                        x_vec_est=x_vec_est,
                        u_vec=np.array([[0.], [0.]]),
                        P_matrix=P_matrix,
                        z_new=z_new
                    )
                    P_matrix = P_matrix_new
            # End of Kalman Filter
            # Start of adding sgt to the dataframe
            
            sgt_vx = (gt_x[1:] - gt_x[:-1]) / 0.05
            sgt_vy = (gt_y[1:] - gt_y[:-1]) / 0.05
            

            df_new['sgt_x'] = gt_x
            df_new['sgt_y'] = gt_y
            df_new['sgt_heading'] = gt_heading
            
            df_new['sgt_vx'] = np.append(np.nan, sgt_vx)
            df_new['sgt_vy'] = np.append(np.nan, sgt_vy)
            df_new['sgt_v'] = np.sqrt(df_new['sgt_vx']**2 + df_new['sgt_vy']**2)
                
            df_new['sgt_ax'] = np.append(np.nan, (df_new['sgt_vx'].values[1:] - df_new['sgt_vx'].values[:-1]) / 0.05)
            df_new['sgt_ay'] = np.append(np.nan, (df_new['sgt_vy'].values[1:] - df_new['sgt_vy'].values[:-1]) / 0.05)
        
            df_to_return = pd.concat([df_to_return, df_new], ignore_index=True)
            
        df_acc = df_to_return
        df_acc.sort_values(by=['scene_id', 'frame_id', 'object_id'], inplace=True)
        ######################################################################
        
        # print(df_acc.to_string())
        # Save Data
        
        df_acc.dropna(inplace=True)
        df_acc['frame_id'] = df_acc['frame_id'] - df_acc['frame_id'].min()
        csv_file_path = os.path.join(output_path, dataset_name, 'data', 'object_pose_motion.csv')
        df_acc.to_csv(csv_file_path, index=False)
        
        # object_id_counts = df_acc['object_id'].value_counts()

        # Filter those that occur 32 times or more
        # frequent_object_ids = object_id_counts[object_id_counts >= 32]

        # Print out the result
        # print(f"Dataset: {dataset_name}, num of objects: {len(frequent_object_ids)}")
    
        ####################### Plot Trajectories #######################
        if plot_estimated_traj:
            plot_poses(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_camera', plot_estimated=True) # Camera
            plot_poses(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_objects_motion', plot_estimated=True) # Motion
            plot_poses(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_objects_poses', plot_estimated=True) # Object estimated poses
            # NuscMini
            # plot_poses(df_nusc, os.path.join(output_path, dataset_name, 'plots'), file_folder='NuScenes_mini', plot_estimated=True) # Nusc
            
        if plot_gt_traj:
            plot_poses(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_camera', plot_gt=True) # Camera
            plot_poses(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_objects_motion', plot_gt=True) # Motion
            plot_poses(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_objects_poses', plot_gt=True) # Object estimated poses
            
        if plot_together_traj:
            plot_poses(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_camera', plot_estimated=True, plot_gt=True) # Camera
            plot_poses(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_objects_motion', plot_estimated=True, plot_gt=True) # Motion
            plot_poses(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_objects_poses', plot_estimated=True, plot_gt=True) # Object estimated poses
            
            
        if plot_xy_pose:
            # plot_eucd_poses(df_nusc, os.path.join(output_path, dataset_name, 'plots'), file_folder='NuScenes_mini', plot_estimated=True) # Nusc
            plot_eucd_poses(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='', plot_estimated=True) # Nusc
            plot_eucd_poses(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='', plot_gt=True) # Nusc
            
            
        ####################### Plot Heading Differences #######################
        if plot_estimated_headings:
            plot_heading_differences(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_camera', plot_estimated=True) # Camera
            plot_heading_differences(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_objects_motion', plot_estimated=True) # Motion
            plot_heading_differences(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_objects_poses', plot_estimated=True) # Object estimated poses
            # NuscMini
            # plot_heading_differences(df_nusc, os.path.join(output_path, dataset_name, 'plots'), file_folder='NuScenes_mini', plot_estimated=True) # Nusc
            
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
            # NuscMini
            # plot_heading_values(df_nusc, os.path.join(output_path, dataset_name, 'plots'), file_folder='NuScenes_mini', plot_estimated=True) # Nusc
            
        if plot_gt_values:
            plot_heading_values(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_camera', plot_gt=True) # Camera
            plot_heading_values(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_objects_motion', plot_gt=True) # Motion
            plot_heading_values(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_objects_poses', plot_gt=True) # Object estimated poses
            
        if plot_together_values:
            plot_heading_values(df_cmr, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_camera', plot_estimated=True, plot_gt=True) # Camera
            plot_heading_values(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_objects_motion', plot_estimated=True, plot_gt=True) # Motion
            plot_heading_values(df_obj_save, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_objects_poses', plot_estimated=True, plot_gt=True) # Object estimated poses
        
         ####################### Plot Velocity Values #######################
        if plot_estimated_velocity:
            plot_velocity_values(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_objects_motion', plot_estimated=True) # Motion
            # NuscMini
            # plot_velocity_values(df_nusc, os.path.join(output_path, dataset_name, 'plots'), file_folder='NuScenes_mini', plot_estimated=True) # Nusc
            
        if plot_gt_velocity:
            plot_velocity_values(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='gt_objects_motion', plot_gt=True) # Motion
            
        if plot_together_velocity:
            plot_velocity_values(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='full_objects_motion', plot_estimated=True, plot_gt=True) # Motion
        
        ####################### Plot CTRV #######################
        if plot_estimated_CTRV:
            plot_ctrv_model(df_acc, os.path.join(output_path, dataset_name, 'plots'), file_folder='est_objects_motion', vis_hist_used=True, save_to_csv=True) # Motion
            
        
        ####################### Plot Heading, (x,y), velocity, acceleration for TRON #######################
        if plot_tron:
            plots_for_tron(df_acc, output_path=os.path.join(output_path, dataset_name, 'plots'), file_folder="TRON_paper", xy=True, heading=True, velocity=True, acceleration=True)

if __name__ == '__main__':
    process_data()
    




