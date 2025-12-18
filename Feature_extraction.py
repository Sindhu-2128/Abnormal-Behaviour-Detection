import cv2
import os
import numpy as np
from weight_matrix import Weight_matrix
from split import Spliter
from getFeatureUV import getFeaturesUV
from poscal import poscal
from labeling import labeling

class Feature_extractor(object):

    def __init__(self, originpics, forgpics, ab_forgpics, U, V, weigh):
        self.originpics = originpics
        self.forgpics = forgpics
        self.ab_forgpics = ab_forgpics
        self.U = U
        self.V = V
        self.weigh = weigh
        self.m = U.shape[0]

    def getPosition(self, img_mask, frame_index):
        """
        --- THIS IS THE FIX ---
        This function now accepts an image/mask directly, not a list of paths.
        --- END FIX ---
        """
        this_Spliter = Spliter()
        
        # We find the path to the abnormal mask using the frame_index
        ab_img_path = self.ab_forgpics[frame_index]
        ab_img = cv2.imread(ab_img_path) if os.path.exists(ab_img_path) else None
        
        # Get initial large blobs from the provided mask
        initial_positions, mopho_img = poscal(img_mask)
        
        # Use the Spliter to break up large blobs
        final_positions = this_Spliter.split(initial_positions, mopho_img, self.weigh)
        
        # Label the final, clean set of positions
        _, label = labeling(final_positions, ab_img)
        
        return final_positions, label

    def get_features_and_labels_with_indices(self, start, end):
        # (This function is for training and remains unchanged)
        all_features = []
        all_labels = []
        all_indices = []
        frame_to_positions = {}

        num_flow_frames = self.U.shape[2]
        limit = min(end - 1, num_flow_frames)
        total_frames_to_process = limit - start
        
        print(f"\n[INFO] Will process {total_frames_to_process} frames...")

        for i in range(start, limit):
            print(f"\r[INFO] Processing frame {i - start + 1}/{total_frames_to_process}...", end="", flush=True)

            # Use the original getPosition that works with file paths for training
            positions, labels = self.getPosition_from_path(self.forgpics, i)

            if positions.size == 0:
                continue

            u_weighted = self.U[:, :, i] * np.sqrt(self.weigh).reshape((self.m, 1))
            v_weighted = self.V[:, :, i] * np.sqrt(self.weigh).reshape((self.m, 1))
            features = getFeaturesUV(positions, u_weighted, v_weighted)

            if features.size > 0:
                all_features.append(features)
                all_labels.append(labels)
                for local_idx in range(features.shape[0]):
                    all_indices.append((i, local_idx))
                
                frame_to_positions[i] = positions
        
        print("\n[INFO] Feature extraction complete.")
        
        if not all_features:
            return np.array([]), np.array([]), [], {}

        final_features = np.nan_to_num(np.concatenate(all_features, axis=0))
        final_labels = np.nan_to_num(np.concatenate(all_labels, axis=0))

        return final_features, final_labels, all_indices, frame_to_positions

    def getPosition_from_path(self, pics, index):
        """ The original getPosition function, renamed to be used by the training script. """
        if not os.path.exists(pics[index]):
            return np.zeros((0, 5)), None
        
        img = cv2.imread(pics[index])
        if img is None:
            return np.zeros((0, 5)), None
        
        return self.getPosition(img, index)