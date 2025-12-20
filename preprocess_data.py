import os
import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# --- CONFIG ---
RAW_TRAIN_DIR = "train"  # Folder containing original train images
BBOX_CSV = "train_bboxes.csv"
OUTPUT_DIR = "processed_data"

def get_quadrant_label(img_name, q_idx, df_bboxes, width, height):
    """
    Determines the class of a quadrant based on bbox centers.
    Returns 'empty' if no bbox center falls in this quadrant.
    """
    # Filter bboxes for this image
    bboxes = df_bboxes[df_bboxes['image_name'] == img_name]
    
    # Define quadrant boundaries
    mid_x, mid_y = width // 2, height // 2
    
    # Q0: TL, Q1: TR, Q2: BL, Q3: BR
    # Check each bbox to see if its center lies in this quadrant
    for _, row in bboxes.iterrows():
        cx = (row['x_min'] + row['x_max']) / 2
        cy = (row['y_min'] + row['y_max']) / 2
        
        q_loc = -1
        if cx < mid_x and cy < mid_y: q_loc = 0
        elif cx >= mid_x and cy < mid_y: q_loc = 1
        elif cx < mid_x and cy >= mid_y: q_loc = 2
        elif cx >= mid_x and cy >= mid_y: q_loc = 3
        
        if q_loc == q_idx:
            return row['class'] # Found the object!
            
    return "empty" # No object center found here

def process_data():
    df = pd.read_csv(BBOX_CSV)
    image_files = [f for f in os.listdir(RAW_TRAIN_DIR) if f.endswith(('.png', '.jpg'))]
    
    # Create directories
    for cls in ['bolt', 'nut', 'washer', 'locatingpin', 'empty']:
        os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)
        
    print(f"Processing {len(image_files)} images...")
    
    for img_file in tqdm(image_files):
        img_path = os.path.join(RAW_TRAIN_DIR, img_file)
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        mid_x, mid_y = w // 2, h // 2
        
        # Crop 4 Quadrants
        # Q0: Top-Left, Q1: Top-Right, Q2: Bot-Left, Q3: Bot-Right
        quadrants = [
            img[0:mid_y, 0:mid_x],   # Q0
            img[0:mid_y, mid_x:w],   # Q1
            img[mid_y:h, 0:mid_x],   # Q2
            img[mid_y:h, mid_x:w]    # Q3
        ]
        
        for q_idx, q_img in enumerate(quadrants):
            label = get_quadrant_label(img_file, q_idx, df, w, h)
            
            # Save file: {parent_name}_{quad_idx}.jpg
            save_name = f"{os.path.splitext(img_file)[0]}_{q_idx}.jpg"
            save_path = os.path.join(OUTPUT_DIR, label, save_name)
            cv2.imwrite(save_path, q_img)
            
    print("✅ Data Prep Complete!")

if __name__ == "__main__":
    process_data()