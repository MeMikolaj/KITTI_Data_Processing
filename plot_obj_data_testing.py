import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

# Path to the CSV file
csv_file_path = "/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/raw/0000/0000.csv"

# Load the CSV data into a DataFrame
df = pd.read_csv(csv_file_path)

# Assuming the CSV has columns: 'object_ID', 'frame_ID', 'x', 'y', 'yaw'
unique_object_ids = df['object_ID'].unique()

# Base directory to save figures
base_directory = "/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/raw/0000"

# Create directory for the object trajectories
folder_name = "test_obj_traj"
folder_path = os.path.join(base_directory, folder_name)

# Create the folder if it does not exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for object_id in unique_object_ids:
    # Filter the DataFrame for the current object ID
    object_data = df[df['object_ID'] == object_id]

    # Sort by frame_ID
    object_data = object_data.sort_values(by='frame_ID')

    # Extract coordinates and yaw
    x = object_data['x'].values
    y = object_data['y'].values
    yaw = object_data['yaw'].values

    for i in tqdm(range(len(x)), "Generating Plots"):
        # Initialize the plot for the current frame
        plt.figure(figsize=(10, 10))

        # Plot the trajectory up to the current frame
        plt.plot(x[:i+1], y[:i+1], label=f'Object ID: {object_id}', color='blue')

        # Plot yaw as an arrow to visualize heading for each frame
        for j in range(i + 1):
            plt.arrow(x[j], y[j], 0.5 * np.cos(yaw[j]), 0.5 * np.sin(yaw[j]),
                      head_width=0.2, head_length=0.2, fc='red', ec='red', alpha=0.5)

        # Customize the plot
        plt.title('Vehicle Trajectory and Heading')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.axis('equal')
        plt.grid()

        # Save the figure for the current frame
        plot_file_path = os.path.join(folder_path, f'plot_object_{object_id}_{object_data.iloc[i]["frame_ID"]}.png')
        plt.savefig(plot_file_path)
        plt.close()  # Close the figure to free memory

print("Finished")
