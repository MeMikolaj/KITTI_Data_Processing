import cv2
import os
import numpy as np
import shutil
from tqdm import tqdm

def draw_boxes_on_image(image_path, pose_data):
    # Load the image
    image = cv2.imread(image_path)
    
    def random_color():
        return tuple(np.random.randint(0, 256, 3).tolist())
    
    for data in pose_data:
        frame, obj_id, x1, y1, x2, y2, _, _, _, _ = data
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Generate a random color
        color = random_color()
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Prepare text and font settings
        text = str(int(obj_id))
        font_scale = 1
        font_thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Calculate text position
        text_x = x1 + 5
        text_y = y1 + text_height + 5
        if text_x + text_width > x2:
            text_x = x2 - text_width - 5
        if text_y > y2:
            text_y = y2 - 5
        
        # Draw text inside the bounding box
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)
    
    return image

def process_folder(base_folder):
    
    objects_found = set() # only draw on the photos when you see the object for the first time
    # Loop through folders in base_folder
    for folder_name in tqdm(os.listdir(base_folder), desc="Processing data folders"):
        folder_path = os.path.join(base_folder, folder_name)
        if not os.path.isdir(folder_path):
            raise Exception(folder_name + " is not a directory.")
        
        # Create 'labeled' folder. If it already exists, remove it first.
        labeled_folder = os.path.join(folder_path, 'labeled_image_0')
        if os.path.exists(labeled_folder):
            shutil.rmtree(labeled_folder)
        os.makedirs(labeled_folder, exist_ok=True)
        
        image_folder = os.path.join(folder_path, 'image_0')
        pose_file = os.path.join(folder_path, 'object_pose.txt')
        
        if not os.path.isdir(image_folder) or not os.path.isfile(pose_file):
            raise Exception(folder_name + " doesn't contain image_0 folder or object_pose.txt file.")
        
        # Read pose data
        with open(pose_file, 'r') as file:
            pose_data = [list(map(float, line.strip().split())) for line in file]
        
        # Loop through images
        for image_file in os.listdir(image_folder):
            if not image_file.lower().endswith('.png'):
                raise Exception(image_file + " is not a png.")
            
            image_path = os.path.join(image_folder, image_file)
            image_name, _ = os.path.splitext(image_file)
            image_frame = int(image_name)
            
            # Filter pose data for the current image frame
            filtered_pose_data = [data for data in pose_data if (int(data[0]) == image_frame and int(data[1]) not in objects_found)]
            
            if len(filtered_pose_data) > 0:
                for data in filtered_pose_data:
                    objects_found.add(int(data[1]))
                    
                # Draw bounding boxes on the image
                image_with_boxes = draw_boxes_on_image(image_path, filtered_pose_data)
                
                # Save the labeled image
                labeled_image_path = os.path.join(labeled_folder, image_file)
                cv2.imwrite(labeled_image_path, image_with_boxes)
        objects_found.clear() # Reset for the next folder

if __name__ == "__main__":
    base_folder = '/home/mikolaj@acfr.usyd.edu.au/datasets/KITTI/raw'
    process_folder(base_folder)
