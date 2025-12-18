# core.py
import cv2
import os
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

from weight_matrix import Weight_matrix
from Feature_extraction import Feature_extractor
from Classifiers import Classifiers

def load_all_datasets():
    """
    Finds and loads data from our specific dataset folders.
    """
    print("[INFO] Loading all datasets...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    datasets_root = os.path.join(base_dir, 'ref_data') # The root is now ref_data

    # --- THIS IS THE KEY MODIFICATION ---
    # We explicitly define the names of our dataset folders.
    dataset_names = ["abuse", "normal"]
    print(f"[INFO] Attempting to load datasets: {dataset_names}")
    # --- END MODIFICATION ---

    all_u_data, all_v_data, all_fg_imgs, all_original_imgs, all_ab_fg_imgs = [], [], [], [], []
    
    for name in dataset_names:
        dataset_dir = os.path.join(datasets_root, name)
        if not os.path.isdir(dataset_dir):
            print(f"[WARNING] Dataset folder not found: {name}. Skipping.")
            continue
            
        frames_dir = os.path.join(dataset_dir, 'frames')
        fg_dir = os.path.join(dataset_dir, 'fg_pics')
        ab_fg_dir = os.path.join(dataset_dir, 'ab_fg_pics')
        flow_path = os.path.join(dataset_dir, 'optical_flow.mat')

        if not os.path.exists(flow_path):
            print(f"[WARNING] Optical flow not found for {name}. Skipping.")
            continue

        print(f"[INFO] Loading data from {name}...")
        flow_data = scipy.io.loadmat(flow_path)
        u_data = flow_data['u_flow']
        v_data = flow_data['v_flow']
        
        num_frames = u_data.shape[2] + 1
        
        all_u_data.append(u_data)
        all_v_data.append(v_data)
        all_original_imgs.extend([os.path.join(frames_dir, f'{i+1:03d}.tif') for i in range(num_frames)])
        all_fg_imgs.extend([os.path.join(fg_dir, f'{i+1:03d}.tif') for i in range(num_frames)])
        all_ab_fg_imgs.extend([os.path.join(ab_fg_dir, f'{i+1:03d}.png') for i in range(num_frames)])

    if not all_u_data:
        print("[ERROR] No valid datasets were loaded. Exiting.")
        return [], [], [], [], [], [], 0

    # We concatenate the flow data along the frame axis (the last axis)
    combined_u = np.concatenate(all_u_data, axis=-1)
    combined_v = np.concatenate(all_v_data, axis=-1)
    
    total_frames = combined_u.shape[2] + 1
    print(f"[INFO] Successfully loaded and combined {len(all_u_data)} datasets.")
    print(f"[INFO] Total frames to process: {total_frames}")

    return combined_u, combined_v, all_fg_imgs, all_original_imgs, all_ab_fg_imgs, datasets_root, total_frames

def main():
    u_data, v_data, fg_imgs, original_imgs, ab_fg_imgs, ref_data_path, num_frames = load_all_datasets()

    if num_frames == 0 or u_data is None:
        print("[CRITICAL] No data loaded. Exiting.")
        return

    frame_height = u_data.shape[0]
    weight = Weight_matrix(ref_data_path=ref_data_path, frame_height=frame_height).get_weight_matrix()

    # The Feature Extractor now gets the combined data from all videos
    thisFeatureExtractor = Feature_extractor(original_imgs, fg_imgs, ab_fg_imgs, u_data, v_data, weight)

    print(f"\n[INFO] Extracting features from all {num_frames} combined frames.")
    all_features, all_labels, all_indices, _ = thisFeatureExtractor.get_features_and_labels_with_indices(0, num_frames)

    # (The rest of the main function remains the same)
    if all_features.size == 0:
        print("[ERROR] No features were extracted.")
        return
    
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"\n[INFO] Total features extracted: {all_features.shape[0]}")
    print(f"[INFO] Overall label distribution: {dict(zip(unique_labels, counts))}")

    if len(unique_labels) < 2:
        print("\n[CRITICAL ERROR] The combined dataset contains only ONE class.")
        return

    feature_indices_to_split = range(len(all_features))
    train_data, test_data, train_labels, test_labels, _, _ = train_test_split(
        all_features, all_labels, feature_indices_to_split, test_size=0.20, random_state=42, stratify=all_labels # Using 20% for test
    )

    print("\n--- DATA SPLIT ---")
    print(f"Training set: {train_data.shape[0]} samples")
    print(f"Testing set:  {test_data.shape[0]} samples")
    print("--- END DATA SPLIT ---\n")

    classifiers = Classifiers(train_data, train_labels)

    # Save the newly trained models
    print("\n[INFO] Saving trained models to disk...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    for name, model in classifiers.models.items():
        filename = os.path.join(models_dir, f'{name}_model.pkl')
        joblib.dump(model, filename)
        print(f"[SUCCESS] Saved {name} model to {filename}")

if __name__ == '__main__':
    main()