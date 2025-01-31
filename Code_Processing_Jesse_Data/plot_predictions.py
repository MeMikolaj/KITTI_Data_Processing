import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib.pyplot as plt
from jesse_utils import *
from natsort import natsorted

###############################################################################################
###############################################################################################  
###############################################################################################  
    
def plot_predictions(df_real, df_prediction, frame_id, ph, output_path, save_png=False, save_pdf=False):
    
    df_trajectron = df_prediction.copy()
    # Trajectron History and Future
    x_trajectron_hist = df_trajectron.loc[df_trajectron['Type'] == 'History', 'x'].values
    y_trajectron_hist = df_trajectron.loc[df_trajectron['Type'] == 'History', 'y'].values
    
    x_trajectron_pred = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'x'].values
    y_trajectron_pred = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'y'].values
    
    # Ground Truth History and Future
    x_gt_hist = df_trajectron.loc[df_trajectron['Type'] == 'History', 'x_actual'].values
    y_gt_hist = df_trajectron.loc[df_trajectron['Type'] == 'History', 'y_actual'].values
    
    x_gt_future = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'x_actual'].values
    y_gt_future = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'y_actual'].values
    
    
    # Plot values
    plt.figure()
    
    ## Histories are the same (*Trajectron uses up to 20 history frames, exactly the same as ground-truth, CTRV uses up to 8) 
    # plt.plot(x_gt_hist, y_gt_hist, linestyle='-',label=f'History used for prediction', color='black')
    
    ######## Plot Whole Trjeactory! History and Future
    df_motion = df_real.copy()
    
    # All gt History Values of an object:
    x_gt_hist_all = df_motion.loc[df_motion['frame_id'] <= int(frame_id), 'x'].values
    y_gt_hist_all = df_motion.loc[df_motion['frame_id'] <= int(frame_id), 'y'].values
    plt.plot(x_gt_hist_all, y_gt_hist_all, ls='-', linewidth=2.0, label=f'DynoSAM Trajectory', color='black')
    
    # All gt Future Values of an object:
    # x_gt_fut_all = df_motion.loc[df_motion['frame_id'] >= int(frame_id), 'x'].values
    # y_gt_fut_all = df_motion.loc[df_motion['frame_id'] >= int(frame_id), 'y'].values
    # plt.plot(x_gt_fut_all, y_gt_fut_all, ls='--', linewidth=2.0, label=f'Ground Truth Future', color='black')
    ########
    
    # Trajectron Prediction
    plt.plot(np.concatenate(([x_gt_hist[-1]], x_trajectron_pred)), np.concatenate(([y_gt_hist[-1]], y_trajectron_pred)), linewidth=2.0, linestyle='--',label=f'Trajectron++ Prediction', color='red')
    
    # Ground-truth Future
    #plt.plot(np.concatenate(([x_gt_hist[-1]], x_gt_future)), np.concatenate(([y_gt_hist[-1]], y_gt_future)), linestyle='--',label=f'Ground-Truth Future', color='blue')
    
    
    # Draw where the object is now, in the future and according to the prediction
    object2 = plt.Circle((x_gt_hist[-1], y_gt_hist[-1]), 0.4, color='orange', label=f'Current location', fill=True, linewidth=2)
    plt.gca().add_artist(object2)
    
    # Draw Last predicted point form the gt data
    # object = plt.Circle((x_gt_fut_all[ph], y_gt_fut_all[ph]), 0.4, color='blueviolet', label=f'Future location after {ph} steps', fill=True, linewidth=2)
    # plt.gca().add_artist(object)
    
    # Trajectron Last predicted point - drawing a circle
    object1 = plt.Circle((x_trajectron_pred[ph-1], y_trajectron_pred[ph-1]), 0.4, color='red', fill=True, linewidth=2)
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
    
    if save_png:
        plot_file_path = os.path.join(output_path, f"frame_{frame_id}.png")
        plt.savefig(plot_file_path, format="png", bbox_inches="tight")
    if save_pdf:
        plot_file_path = os.path.join(output_path, f"frame_{frame_id}.pdf")
        plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory
    
    # Computer ADE and FDE
    ade_trajectron = 0; fde_trajectron = 0
    if True:
        for i in range(len(x_gt_future)):
            ade_trajectron += np.sqrt((x_trajectron_pred[i] - x_gt_future[i]) ** 2 + (y_trajectron_pred[i] - y_gt_future[i]) ** 2)
            
        ade_trajectron /= ph
        fde_trajectron = np.sqrt((x_trajectron_pred[-1] - x_gt_future[-1]) ** 2 + (y_trajectron_pred[-1] - y_gt_future[-1]) ** 2)
    
    return ade_trajectron, fde_trajectron, (x_trajectron_pred[-1], y_trajectron_pred[-1]), (x_trajectron_pred[-2], y_trajectron_pred[-2])

##############################################################
def plot_pred_euc(distance_arr, output_path, save_png=False, save_pdf=False):
    
    # subarrays = np.array_split(distance_arr, 3)
    # plt.bar(np.arange(1, len(subarrays[0]) + 1), height=subarrays[0], width=0.33, color='red', label=f"Estimated, avg: {round(sum(subarrays[0])/len(subarrays[0]), 2)}")
    # plt.bar(0.34+np.arange(1, len(subarrays[1]) + 1), height=subarrays[1], width=0.33, color='green', label=f"Ground Truth, avg: {round(sum(subarrays[1])/len(subarrays[1]), 2)}")
    # plt.bar(0.67+np.arange(1, len(subarrays[2]) + 1), height=subarrays[2], width=0.33, color='blue', label=f"Ground Truth + EKF, avg: {round(sum(subarrays[2])/len(subarrays[2]), 2)}")
    
    data = np.array(distance_arr)
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    plt.bar(np.arange(1, len(normalized_data) + 1), height=normalized_data, color='red', label=f"Estimated, avg: {round(sum(normalized_data)/len(normalized_data), 2)}")
    #plt.axhline(y=sum(distance_arr)/len(distance_arr), color='red', linestyle='--', label=f'Average = {round(sum(distance_arr)/len(distance_arr), 3)}')
    
    plt.xticks(fontsize=20)  # Change x-axis tick label size
    plt.yticks(fontsize=20)  # Change y-axis tick label size
    plt.xlabel('Consecutive Frames', fontsize=22)
    plt.ylabel('Absolute Consistency Error', fontsize=22)
    plt.legend(loc='best', fontsize=12)
    # plt.axis('equal')
    plt.grid()
    
    if save_png:
        plot_file_path = os.path.join(output_path, f"ConsistencyError_nusc.png")
        plt.savefig(plot_file_path, format="png", bbox_inches="tight")
    if save_pdf:
        plot_file_path = os.path.join(output_path, f"ConsistencyError_nusc.pdf")
        plt.savefig(plot_file_path, format="pdf", bbox_inches="tight")
    plt.close()

    
def calculate_ATE(df_prediction, ph):
    
    df_trajectron = df_prediction.copy()
    
    x_trajectron_pred = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'x'].values
    y_trajectron_pred = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'y'].values
    
    # Ground Truth History and Future
    x_gt_hist = df_trajectron.loc[df_trajectron['Type'] == 'History', 'x_actual'].values
    y_gt_hist = df_trajectron.loc[df_trajectron['Type'] == 'History', 'y_actual'].values
    
    x_gt_future = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'x_actual'].values
    y_gt_future = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'y_actual'].values
    heading_gt_future = df_trajectron.loc[df_trajectron['Type'] == 'Future', 'heading_actual'].values
    
    
    ###########################
    errors_t = []
    errors_r = []
    # ATE
    # for i in range(ph):
    i = 29
    if i==0:
        theta_pred = np.arctan2(y_trajectron_pred[i] - y_gt_hist[-1], x_trajectron_pred[i]-x_gt_hist[-1])
    else:
        theta_pred = np.arctan2(y_trajectron_pred[i] - y_trajectron_pred[i-1], x_trajectron_pred[i]-x_trajectron_pred[i-1])

    pred_pose = np.eye(3, dtype=np.float64)
    pred_pose[0, 2] = x_trajectron_pred[i]
    pred_pose[1, 2] = y_trajectron_pred[i]
    pred_pose[0:2, 0:2] = np.array([[np.cos(theta_pred), -np.sin(theta_pred)], [np.sin(theta_pred), np.cos(theta_pred)]])

    gt_heading = heading_gt_future[i]
    gt_pose = np.eye(3, dtype=np.float64)
    gt_pose[0, 2] = x_gt_future[i]
    gt_pose[1, 2] = y_gt_future[i]
    gt_pose[0:2, 0:2] = np.array([[np.cos(gt_heading), -np.sin(gt_heading)], [np.sin(gt_heading), np.cos(gt_heading)]])

    se2_error = np.linalg.inv(gt_pose) @ pred_pose
    error_t = np.linalg.norm(se2_error[:, 2])
    errors_t.append(error_t)

    trace_r = se2_error[0,0] + se2_error[1,1]
    cos_theta = (trace_r - 1)/2
    
    error_r = np.arccos(cos_theta)
    errors_r.append(error_r)

    return errors_t, errors_r

        
 
###############################################################################################
###############################################################################################  
###############################################################################################
 

base_path = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/Jesse_processed'

datasets = ['0061']#, '0757']#['0000', '0003', '0005', '0018', '0020'] # '0061', '0757'
estimation_methods = ['est']#, 'gt', 'sgt']

def process():
    startup_plotting(font_size=16)
    
    for dataset_name in datasets:
        data_path   = os.path.join(base_path, dataset_name, 'data', 'object_pose_motion.csv')
        df_dataset = pd.read_csv(data_path)

        print("*****************************************")
        euc_pred_distance = []
        for estimation_method in estimation_methods:

            df_real = df_dataset.copy()
            ade_error = 0; fde_error = 0; avg_consistency_error = 0; error_counter = 0; error_counter_cons = 0 # Evaluation Errors
        
            objects_path = os.path.join(base_path, dataset_name, 'data', 'predictions_global', estimation_method)
            object_folders = os.listdir(objects_path) # List Objects
        
            # Filter dataframe with GT trajectory
            if estimation_method == 'est':
                df_real = df_real[['object_id', 'frame_id', 'x', 'y']]
            elif estimation_method == 'gt':
                df_real = df_real[['object_id', 'frame_id', 'gt_x', 'gt_y']]
                df_real = df_real.rename(columns={'gt_x': 'x', 'gt_y': 'y'})
            elif estimation_method == 'sgt':
                df_real = df_real[['object_id', 'frame_id', 'sgt_x', 'sgt_y']]
                df_real = df_real.rename(columns={'sgt_x': 'x', 'sgt_y': 'y'})
            else:
                raise Exception("Estimation methods must be est, gt or sgt")
            
            errors_t = []
            errors_r = []
            
            for object_name in object_folders: # tqdm(object_folders, desc="creating prediction plots for objects"):

                last_pred = None # Last prediction for ACE

                

                df_real_obj = df_real.copy()
                df_real_obj = df_real_obj[df_real_obj['object_id'] == object_name]
                
                # if object_name != "32a":
                #     continue

                predictions_path = os.path.join(objects_path, object_name)
                
                # Make sure hidden files are not considered (in case some csv was opened rn)
                all_files = os.listdir(predictions_path)
                filtered_files = [f for f in all_files if not f.startswith('.')]
                predictions_data_files = natsorted(filtered_files)    
            
                for prediction_file_name in predictions_data_files:
            
                    prediction_file_path = os.path.join(predictions_path, prediction_file_name)
                    df_prediction = pd.read_csv(prediction_file_path)

                    plots_path = os.path.join(base_path, dataset_name, 'plots', object_name, 'traj_predictions', estimation_method)
                    maybe_makedirs(plots_path)

                    frame_id = prediction_file_name.split('.')[0]

                    
                    # Plot predictions - Trajectron only
                    ade, fde, last, prev_to_last = plot_predictions(df_real_obj, df_prediction, frame_id, ph=30, output_path=plots_path, save_png=True, save_pdf=True)
                    
                    error_t, error_r = calculate_ATE(df_prediction, ph=30)
                    
                    if last_pred == None:
                        last_pred = last
                    else:
                        avg_consistency_error += np.linalg.norm(np.array(last_pred) - np.array(prev_to_last))
                        error_counter_cons    += 1

                        euc_pred_distance.append(np.linalg.norm(np.array(last_pred) - np.array(prev_to_last)))
    
                        last_pred = last

                    # Add errors
                    ade_error             += ade
                    fde_error             += fde
                    error_counter         += 1
                    
                    errors_t.append(error_t)
                    errors_r.append(error_r)
                    
            ade_error /= error_counter
            fde_error /= error_counter
            avg_consistency_error /= error_counter_cons
            
            rmse_t_length = len(errors_t)
            rmse_t = np.sqrt(np.sum(np.square(errors_t))/rmse_t_length)
            
            rmse_r_length = len(errors_r)
            rmse_r = np.sqrt(np.sum(np.square(errors_r))/rmse_r_length)
            
            
            print(f"Dataset: {dataset_name}, Method: {estimation_method}")
            # print(f"Results: ADE: {round(ade_error, 2)}, FDE: {round(fde_error, 2)}, ACE: {round(avg_consistency_error, 2)}")
            print(f"Results: RMSE_t: {round(rmse_t, 2)}, RMSE_r: {round(rmse_r, 2)}")
            print("---------------------------------------------")
        # plot_pred_euc(euc_pred_distance, os.path.join(base_path), save_png=True, save_pdf=True)
if __name__ == '__main__':
    process()