# generate_optical_flow.py
import cv2
import os
import numpy as np
import scipy.io
import glob
import time
import argparse # We use argparse to accept command-line arguments

def generate_for_dataset(dataset_name):
    """
    Calculates dense optical flow for a specific dataset folder.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    
    dataset_dir = os.path.join(base_dir, 'ref_data',  dataset_name)
    image_dir = os.path.join(dataset_dir, 'original_pics')
    output_path = os.path.join(dataset_dir, 'optical_flow.mat') # Save directly in the dataset folder
    
    if not os.path.isdir(image_dir):
        print(f"[ERROR] Image directory not found: {image_dir}")
        return

    print(f"\n--- Processing Dataset: {dataset_name} ---")
    print(f"[INFO] Image source: {image_dir}")
    print(f"[INFO] Output file: {output_path}")

    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
    
    num_frames = len(image_paths)
    if num_frames < 2:
        print("[ERROR] Need at least 2 images.")
        return
        
    print(f"[INFO] Found {num_frames} images.")

    frame_prev = cv2.imread(image_paths[0])
    frame_prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY)
    
    u_frames = []
    v_frames = []
    start_time = time.time()
    
    for i in range(1, num_frames):
        print(f"\r[INFO] Calculating flow for frame {i}/{num_frames - 1}...", end="", flush=True)
        
        frame_next = cv2.imread(image_paths[i])
        if frame_next is None: continue
        frame_next_gray = cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(frame_prev_gray, frame_next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        u_frames.append(flow[..., 0])
        v_frames.append(flow[..., 1])
        frame_prev_gray = frame_next_gray
        
    print("\n[INFO] Optical flow calculation complete.")
    
    u_stack = np.stack(u_frames, axis=-1)
    v_stack = np.stack(v_frames, axis=-1)
    
    print(f"[INFO] Final data shape: {u_stack.shape}")
    
    print(f"[INFO] Saving combined flow to {output_path}...")
    scipy.io.savemat(output_path, {'u_flow': u_stack, 'v_flow': v_stack})
    
    end_time = time.time()
    print(f"[SUCCESS] Finished in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    # --- This part allows you to specify the folder from the command line ---
    parser = argparse.ArgumentParser(description="Generate optical flow for a dataset.")
    parser.add_argument("dataset_name", type=str, help="The name of the dataset folder inside ref_data/datasets/ (e.g., 'video1_church')")
    args = parser.parse_args()
    
    generate_for_dataset(args.dataset_name)