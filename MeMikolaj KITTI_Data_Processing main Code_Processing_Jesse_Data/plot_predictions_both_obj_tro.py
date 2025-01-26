import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib.pyplot as plt
from jesse_utils import *

###############################################################################################
###############################################################################################  
###############################################################################################  
    
def plot_predictions(df_ctrv, df_trajectron, df_trajectron2, output_path, frame_id, frame_id2, object_name, dataset_name, ph, h, dt, ade_ctrv, ade_trajectron, fde_ctrv, fde_trajectron, include_ctrv=True, save=True):
    
    # CTRV History and Future
    x_ctrv_hist = df_ctrv.loc[df_ctrv['Type'] == 'History', 'x'].values
    y_ctrv_hist = df_ctrv.loc[df_ctrv['Type'] == 'History', 'y'].values
    
    x_ctrv_pred = df_ctrv.loc[df_ctrv['Type'] == 'Future', 'x'].values
    y_ctrv_pred = df_ctrv.loc[df_ctrv['Type'] == 'Future', 'y'].values
    
    
    # Trajectron History and Future
    x_trajectron_hist = df_trajectron.loc[df_trajectron['Type'] == 'History', 'x'].values
    y_trajectron_hist = df_trajectron.loc[df_trajectron['Type'] == 'History', 'y'].values
    
    x_trajectron_pred = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'x'].values
    y_trajectron_pred = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'y'].values
    
    x_trajectron_pred2 = df_trajectron2.loc[df_trajectron2['Type'] == 'Future', 'x'].values
    y_trajectron_pred2 = df_trajectron2.loc[df_trajectron2['Type'] == 'Future', 'y'].values
    
    # Ground Truth History and Future
    x_gt_hist = df_trajectron.loc[df_trajectron['Type'] == 'History', 'x_visual'].values
    y_gt_hist = df_trajectron.loc[df_trajectron['Type'] == 'History', 'y_visual'].values
    
    x_gt_future = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'x_visual'].values
    y_gt_future = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'y_visual'].values
    
    x_gt_hist2 = df_trajectron2.loc[df_trajectron2['Type'] == 'History', 'x_visual'].values
    y_gt_hist2 = df_trajectron2.loc[df_trajectron2['Type'] == 'History', 'y_visual'].values
    
    x_gt_future2 = df_trajectron2.loc[df_trajectron2['Type'] == 'Future', 'x_visual'].values
    y_gt_future2 = df_trajectron2.loc[df_trajectron2['Type'] == 'Future', 'y_visual'].values
    
    
    # Plot values
    plt.figure(figsize=(12, 5))
    
    ## Histories are the same (*Trajectron uses up to 20 history frames, exactly the same as ground-truth, CTRV uses up to 8) 
    # plt.plot(x_gt_hist, y_gt_hist, linestyle='-',label=f'History used for prediction', color='black')
    
    ######## Plot Whole Trjeactory! History and Future
    df_motion = pd.read_csv("/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_processed/0000/data/object_pose_motion.csv")
    df_motion = df_motion[df_motion['object_id'] == '1a']
    
    # All gt History Values of an object:
    x_gt_hist_all = df_motion.loc[df_motion['frame_id'] <= int(frame_id), 'x'].values
    y_gt_hist_all = df_motion.loc[df_motion['frame_id'] <= int(frame_id), 'y'].values
    plt.plot(x_gt_hist_all, y_gt_hist_all, ls='-', linewidth=2.0, label=f'DynoSAM Trajectory', color='black')
    
    # All gt Future Values of an object:
    x_gt_fut_all = df_motion.loc[df_motion['frame_id'] >= int(4), 'x'].values
    y_gt_fut_all = df_motion.loc[df_motion['frame_id'] >= int(4), 'y'].values
    plt.plot(x_gt_fut_all, y_gt_fut_all, ls='--', linewidth=2.0, label=f'Ground Truth Future', color='black')
    
    df_motion2 = pd.read_csv("/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_processed/0000/data/object_pose_motion.csv")
    df_motion2 = df_motion2[df_motion2['object_id'] == '2a']
    
    # All gt History Values of an object:
    x_gt_hist_all2 = df_motion2.loc[df_motion2['frame_id'] <= int(frame_id2), 'x'].values
    y_gt_hist_all2 = df_motion2.loc[df_motion2['frame_id'] <= int(frame_id2), 'y'].values
    plt.plot(x_gt_hist_all2, y_gt_hist_all2, ls='-', linewidth=2.0, color='black')
    
    # All gt Future Values of an object:
    x_gt_fut_all2 = df_motion2.loc[df_motion2['frame_id'] >= int(4), 'x'].values
    y_gt_fut_all2 = df_motion2.loc[df_motion2['frame_id'] >= int(4), 'y'].values
    plt.plot(x_gt_fut_all2, y_gt_fut_all2, ls='--', linewidth=2.0, color='black')
    ########
    
    if include_ctrv:
        # CTRV Prediction
        plt.plot(np.concatenate(([x_gt_hist[-1]], x_ctrv_pred)), np.concatenate(([y_gt_hist[-1]], y_ctrv_pred)), linestyle='--',label=f'CTRV Prediction', color='green')

    # Trajectron Prediction
    plt.plot(np.concatenate(([x_gt_hist[-1]], x_trajectron_pred)), np.concatenate(([y_gt_hist[-1]], y_trajectron_pred)), linewidth=2.0, linestyle='--',label=f'Prediction', color='red')
    
    plt.plot(np.concatenate(([x_gt_hist2[-1]], x_trajectron_pred2)), np.concatenate(([y_gt_hist2[-1]], y_trajectron_pred2)), linewidth=2.0, linestyle='--', color='red')
    
    # Ground-truth Future
    #plt.plot(np.concatenate(([x_gt_hist[-1]], x_gt_future)), np.concatenate(([y_gt_hist[-1]], y_gt_future)), linestyle='--',label=f'Ground-Truth Future', color='blue')
    
    
    # Draw where the object is now, in the future and according to the prediction
    object2 = plt.Circle((x_gt_hist[-1], y_gt_hist[-1]), 0.4, color='dodgerblue', label=f'Current location: object 1', fill=True, linewidth=2, zorder=6)
    plt.gca().add_artist(object2)
    
    object2_2 = plt.Circle((x_gt_hist2[-1], y_gt_hist2[-1]), 0.4, color='limegreen', label=f'Current location: object 2', fill=True, linewidth=2, zorder=6)
    plt.gca().add_artist(object2_2)
    
    # Draw Last predicted point form the gt data
    object = plt.Circle((x_gt_fut_all[ph + int(frame_id)-4], y_gt_fut_all[ph + int(frame_id)-4]), 0.4, color='blueviolet', label=f'Future location after {ph} steps', fill=True, linewidth=2, zorder=5)
    plt.gca().add_artist(object)
    
    object_2 = plt.Circle((x_gt_fut_all2[ph + int(frame_id2)-4], y_gt_fut_all2[ph + int(frame_id2)-4]), 0.4, color='blueviolet', fill=True, linewidth=2, zorder=5)
    plt.gca().add_artist(object_2)
    
    # Trajectron Last predicted point - drawing a circle
    object1 = plt.Circle((x_trajectron_pred[ph-1], y_trajectron_pred[ph-1]), 0.4, color='red', label=f'Predicted location after {ph} steps', fill=True, linewidth=2, zorder=7)
    plt.gca().add_artist(object1)
    
    object1_2 = plt.Circle((x_trajectron_pred2[ph-1], y_trajectron_pred2[ph-1]), 0.4, color='red', fill=True, linewidth=2, zorder=7)
    plt.gca().add_artist(object1_2)
    
    #plt.title(f'Predictions of object: {object_name}, at frame: {frame_id}, dataset: KITTI-{dataset_name}, ph: {ph} steps, h: up to {h} steps, dt: {dt}')
    plt.xticks(fontsize=20)  # Change x-axis tick label size
    plt.yticks(fontsize=20)  # Change y-axis tick label size
    plt.xlabel('X (m)', fontsize=26)
    plt.ylabel('Y (m)', fontsize=26)
    plt.legend(loc='upper right', fontsize=10)
    # plt.legend(loc='lower left', fontsize=18)
    plt.axis('equal')
    plt.grid()
    
    # Set axis limits
    plt.xlim(0, 80)
    plt.ylim(-5, 25)
    
    if save:
        plot_file_path = os.path.join(output_path, f"frame_{frame_id2}.png")
        plt.savefig(plot_file_path, format="png", bbox_inches="tight")
        plot_file_path = os.path.join(output_path, f"frame_{frame_id2}.pdf")
        plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory
    
    # Computer ADE and FDE
    ade_ctrv = 0; ade_trajectron = 0; fde_ctrv = 0; fde_trajectron = 0
    if True:
        for i in range(len(x_gt_future)):
            # ade_ctrv       += np.sqrt((x_ctrv_pred[i] - x_gt_future[i]) ** 2 + (y_ctrv_pred[i] - y_gt_future[i]) ** 2)
            ade_ctrv       += np.sqrt((x_trajectron_pred2[i] - x_gt_future2[i]) ** 2 + (y_trajectron_pred2[i] - y_gt_future2[i]) ** 2)
            ade_trajectron += np.sqrt((x_trajectron_pred[i] - x_gt_future[i]) ** 2 + (y_trajectron_pred[i] - y_gt_future[i]) ** 2)
            
        ade_ctrv       /= ph
        ade_trajectron /= ph
        
        # fde_ctrv       = np.sqrt((x_ctrv_pred[-1] - x_gt_future[-1]) ** 2 + (y_ctrv_pred[-1] - y_gt_future[-1]) ** 2)
        fde_ctrv       = np.sqrt((x_trajectron_pred2[-1] - x_gt_future2[-1]) ** 2 + (y_trajectron_pred2[-1] - y_gt_future2[-1]) ** 2)
        fde_trajectron = np.sqrt((x_trajectron_pred[-1] - x_gt_future[-1]) ** 2 + (y_trajectron_pred[-1] - y_gt_future[-1]) ** 2)
    
    return(ade_ctrv, ade_trajectron, fde_ctrv, fde_trajectron)


###############################################################################################
###############################################################################################  
###############################################################################################
 

base_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_processed'
datasets = ['0000']

def process():
    startup_plotting()
    ade_ctrv = 0; ade_trajectron = 0; fde_ctrv = 0; fde_trajectron = 0; data_counter=0
    for dataset_name in datasets:
        general_path = os.path.join(base_path, dataset_name)
        
        # CTRV DATA
        ctrv_path = os.path.join(general_path, 'data', 'CTRV_data')
        
        # TRAJECTRON DATA
        trajectron_path = os.path.join(general_path, 'data', 'Trajectron_data_destandardized')
        
        object_folders = os.listdir(trajectron_path) # Same for CTRV and TRAJECTRON
        
        # Loop through the objects
        #for object_name_folder in tqdm(object_folders, "creating prediction plots for objects"):
        
        path_to_data = os.path.join(trajectron_path, '1a')
        prediction_data_files = os.listdir(path_to_data)
        
        path_to_data2 = os.path.join(trajectron_path, '2a')
        prediction_data_files2 = os.listdir(path_to_data2)
        
        # path_to_data2 = os.path.join(trajectron_path, object_name_folder)
        # prediction_data_files = os.listdir(path_to_data)

        # Loop through the files with predictions for both CTRV and TRAJECTRON and plot
        for prediction_file in prediction_data_files2:
            # CTRV csv
            # ctrv_file_path = os.path.join(path_to_data, prediction_file)
            # ctrv_df = pd.read_csv(ctrv_file_path)
            
            # TRAJECTRON csv
            file_name = prediction_file.split('_')[1]
            file_number = file_name.split('.')[0]
            if int(file_number) <= 35:
                trajectron_file_path = os.path.join(trajectron_path, '1a', prediction_file)
                trajectron_df = pd.read_csv(trajectron_file_path)
            else:
                trajectron_file_path = os.path.join(trajectron_path, '1a', 'frame_35.csv')
                trajectron_df = pd.read_csv(trajectron_file_path)
            
            trajectron_file_path2 = os.path.join(trajectron_path, '2a', prediction_file)
            trajectron_df2 = pd.read_csv(trajectron_file_path2)
            
            
            #plots_path = os.path.join(general_path, 'plots', object_name_folder, 'ctrv_and_trajectron') # 1
            plots_path = os.path.join(general_path, 'plots', 'trajectron', 'trajectron_both')           # 2
            maybe_makedirs(plots_path)
            frame_id2 = prediction_file.split('_')[1].split('.')[0]
            
            frame_id = frame_id2
            if int(frame_id2) > 35:
                frame_id = '35'
            
            # Plot predictions - 1. CTRV and Trajectron,  2. Trjaectron only
            #results = plot_predictions(ctrv_df, trajectron_df, plots_path, frame_id, object_name_folder, dataset_name, ph=30, h=4, dt=0.05, ade_ctrv=ade_ctrv, ade_trajectron=ade_trajectron, fde_ctrv=fde_ctrv, fde_trajectron=fde_trajectron, include_ctrv=True)
            results = plot_predictions(trajectron_df, trajectron_df, trajectron_df2, plots_path, frame_id, frame_id2, '2a', dataset_name, ph=30, h=4, dt=0.05, ade_ctrv=ade_ctrv, ade_trajectron=ade_trajectron, fde_ctrv=fde_ctrv, fde_trajectron=fde_trajectron, include_ctrv=False, save=True)
            
            # Add errors
            ade_ctrv       += results[0]
            ade_trajectron += results[1]
            fde_ctrv       += results[2]
            fde_trajectron += results[3]
            data_counter   += 1

    ade_ctrv       /= data_counter
    ade_trajectron /= data_counter
    fde_ctrv       /= data_counter
    fde_trajectron /= data_counter
    
    print("---------------------------------------------")
    print(f"Trajectron, object2   -    ADE: {ade_ctrv}, FDE: {fde_ctrv}")
    print("---------------------------------------------")
    print(f"Trajectron, object1   -    ADE: {ade_trajectron}, FDE: {fde_trajectron}")
    print("---------------------------------------------")
        
        
if __name__ == '__main__':
    process()