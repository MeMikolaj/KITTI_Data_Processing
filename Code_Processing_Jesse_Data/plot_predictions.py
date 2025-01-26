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
    
def plot_predictions(df_trajectron, output_path, frame_id, object_name, dataset_name, ph, h, dt, ade_trajectron, fde_trajectron, save=True):
    
    # Trajectron History and Future
    x_trajectron_hist = df_trajectron.loc[df_trajectron['Type'] == 'History', 'x'].values
    y_trajectron_hist = df_trajectron.loc[df_trajectron['Type'] == 'History', 'y'].values
    
    x_trajectron_pred = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'x'].values
    y_trajectron_pred = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'y'].values
    
    # Ground Truth History and Future
    x_gt_hist = df_trajectron.loc[df_trajectron['Type'] == 'History', 'x_visual'].values
    y_gt_hist = df_trajectron.loc[df_trajectron['Type'] == 'History', 'y_visual'].values
    
    x_gt_future = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'x_visual'].values
    y_gt_future = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'y_visual'].values
    
    
    # Plot values
    plt.figure(figsize=(12, 5))
    
    ## Histories are the same (*Trajectron uses up to 20 history frames, exactly the same as ground-truth, CTRV uses up to 8) 
    # plt.plot(x_gt_hist, y_gt_hist, linestyle='-',label=f'History used for prediction', color='black')
    
    ######## Plot Whole Trjeactory! History and Future
    df_motion = pd.read_csv("/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_processed/0000/data/object_pose_motion.csv")
    df_motion = df_motion[df_motion['object_id'] == object_name]
    
    # All gt History Values of an object:
    x_gt_hist_all = df_motion.loc[df_motion['frame_id'] <= int(frame_id), 'x'].values
    y_gt_hist_all = df_motion.loc[df_motion['frame_id'] <= int(frame_id), 'y'].values
    plt.plot(x_gt_hist_all, y_gt_hist_all, ls='-', linewidth=2.0, label=f'DynoSAM Trajectory', color='black')
    
    # All gt Future Values of an object:
    x_gt_fut_all = df_motion.loc[df_motion['frame_id'] >= int(frame_id), 'x'].values
    y_gt_fut_all = df_motion.loc[df_motion['frame_id'] >= int(frame_id), 'y'].values
    plt.plot(x_gt_fut_all, y_gt_fut_all, ls='--', linewidth=2.0, label=f'Ground Truth Future', color='black')
    ########
    
    # Trajectron Prediction
    plt.plot(np.concatenate(([x_gt_hist[-1]], x_trajectron_pred)), np.concatenate(([y_gt_hist[-1]], y_trajectron_pred)), linewidth=2.0, linestyle='--',label=f'Trajectron++ Prediction', color='red')
    
    # Ground-truth Future
    #plt.plot(np.concatenate(([x_gt_hist[-1]], x_gt_future)), np.concatenate(([y_gt_hist[-1]], y_gt_future)), linestyle='--',label=f'Ground-Truth Future', color='blue')
    
    
    # Draw where the object is now, in the future and according to the prediction
    object2 = plt.Circle((x_gt_hist[-1], y_gt_hist[-1]), 0.4, color='orange', label=f'Current location', fill=True, linewidth=2)
    plt.gca().add_artist(object2)
    
    # Draw Last predicted point form the gt data
    object = plt.Circle((x_gt_fut_all[ph], y_gt_fut_all[ph]), 0.4, color='blueviolet', label=f'Future location after {ph} steps', fill=True, linewidth=2)
    plt.gca().add_artist(object)
    
    # Trajectron Last predicted point - drawing a circle
    object1 = plt.Circle((x_trajectron_pred[ph-1], y_trajectron_pred[ph-1]), 0.4, color='red', label=f'Predicted location after {ph} steps', fill=True, linewidth=2)
    plt.gca().add_artist(object1)
    
    
    #plt.title(f'Predictions of object: {object_name}, at frame: {frame_id}, dataset: KITTI-{dataset_name}, ph: {ph} steps, h: up to {h} steps, dt: {dt}')
    plt.xticks(fontsize=20)  # Change x-axis tick label size
    plt.yticks(fontsize=20)  # Change y-axis tick label size
    plt.xlabel('X (m)', fontsize=26)
    plt.ylabel('Y (m)', fontsize=26)
    plt.legend(loc='best', fontsize=12)
    plt.axis('equal')
    plt.grid()
    
    # Set axis limits
    # plt.xlim(27, 71)
    # plt.ylim(2.5, 20)
    
    if save:
        plot_file_path = os.path.join(output_path, f"frame_{frame_id}.png")
        plt.savefig(plot_file_path, format="png", bbox_inches="tight")
        # plot_file_path = os.path.join(output_path, f"frame_{frame_id}.pdf")
        # plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory
    
    # Computer ADE and FDE
    ade_trajectron = 0; fde_trajectron = 0
    if True:
        for i in range(len(x_gt_future)):
            ade_trajectron += np.sqrt((x_trajectron_pred[i] - x_gt_future[i]) ** 2 + (y_trajectron_pred[i] - y_gt_future[i]) ** 2)
            
        ade_trajectron /= ph
        fde_trajectron = np.sqrt((x_trajectron_pred[-1] - x_gt_future[-1]) ** 2 + (y_trajectron_pred[-1] - y_gt_future[-1]) ** 2)
    
    return(ade_trajectron, fde_trajectron)


###############################################################################################
###############################################################################################  
###############################################################################################
 

base_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_processed'
datasets = ['0000']

def process():
    startup_plotting()
    ade_trajectron = 0; fde_trajectron = 0; data_counter=0
    for dataset_name in datasets:
        general_path = os.path.join(base_path, dataset_name)
        
        # TRAJECTRON DATA
        trajectron_path = os.path.join(general_path, 'data', 'Trajectron_data_destandardized')
        
        object_folders = os.listdir(trajectron_path) # Same for CTRV and TRAJECTRON
        
        # Loop through the objects
        last_pred = None
        for object_name_folder in tqdm(object_folders, "creating prediction plots for objects"):
            path_to_data = os.path.join(trajectron_path, object_name_folder)
            prediction_data_files = os.listdir(path_to_data)
            
            if object_name_folder != "2a":
                continue
            
            # predictions = [(0, 0) for _ in range(115)]
            # Loop through the files with predictions for both CTRV and TRAJECTRON and plot
            for prediction_file in prediction_data_files:

                # TRAJECTRON csv
                trajectron_file_path = os.path.join(trajectron_path, object_name_folder, prediction_file)
                trajectron_df = pd.read_csv(trajectron_file_path)
                
                plots_path = os.path.join(general_path, 'plots', object_name_folder, 'est_trajectron')
                    
                maybe_makedirs(plots_path)
                frame_id = prediction_file.split('_')[1].split('.')[0]
                
                # Plot predictions - Trajectron only
                results = plot_predictions(trajectron_df, plots_path, frame_id, object_name_folder, dataset_name, ph=30, h=4, dt=0.05, ade_trajectron=ade_trajectron, fde_trajectron=fde_trajectron, save=True)
                
                # Add errors
                ade_trajectron += results[0]
                fde_trajectron += results[1]
                data_counter   += 1
                
                # predictions[int(frame_id)-4] = ((trajectron_df.loc[trajectron_df['Type'] == 'Future', 'x'].values[29], trajectron_df.loc[trajectron_df['Type'] == 'Future', 'y'].values[29]))
                
            # Plot euclidean Distance
            # euclidean_distance = []
            # for i in range(len(predictions)-1):
            #     euclidean_distance.append(np.sqrt((predictions[i+1][0] - predictions[i][0])**2 + (predictions[i+1][1] - predictions[i][1])**2))
                
            # plt.figure(figsize=(12, 6))

            # x_values = np.arange(len(euclidean_distance))
            # plt.plot(x_values, euclidean_distance, label=f'Euclidean Distance Between Predictions', color='black')
            # mean_euclidean_d = sum(euclidean_distance) / len(euclidean_distance)
            # plt.axhline(y=mean_euclidean_d, linestyle='--', linewidth=2, label=f'Avg Euclidean Distance Between Predictions: {round(mean_euclidean_d, 3)}', color='magenta')

            # plt.title(f'Euclidean Distance between Predictions, est kitti0000, object: 2a')
            # plt.xlabel('Consecutive Frames')
            # plt.ylabel('Euclidean Distance')
            # plt.legend()
            # plt.grid()
            # plot_file_path = os.path.join(general_path, 'plots', object_name_folder, "est_Euclidean_distance_predictions.png")
            # plt.savefig(plot_file_path)
            # plt.close()  # Close the figure to free memory
            # print(predictions)
                

    ade_trajectron /= data_counter
    fde_trajectron /= data_counter
    
    print("---------------------------------------------")
    print(f"Trajectron   -    ADE: {ade_trajectron}, FDE: {fde_trajectron}")
    print("---------------------------------------------")
        
        
if __name__ == '__main__':
    process()