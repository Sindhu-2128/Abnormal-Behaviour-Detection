import cv2
import scipy.io
import numpy as np
from weight_matrix import *
from split import *

# This font variable is from the test function, can be removed if not needed elsewhere
font=cv2.FONT_HERSHEY_COMPLEX

def getFeaturesUV(realPos, u, v):
    """
    Calculates optical flow features (u, v) for each bounding box in realPos.
    Includes a safety check to prevent warnings and NaN values from empty slices.
    """
    # First, check if there are any positions to process at all.
    if realPos.size == 0:
        return np.zeros((0, 2))

    n = realPos.shape[0]
    data = np.zeros((n, 2))
    for i in range(n):
        # Ensure bounding box coordinates can form a valid slice.
        if realPos[i][0] <= realPos[i][1]:
            realPos[i][0] = realPos[i][1] + 1
        if realPos[i][2] <= realPos[i][3]:
            realPos[i][2] = realPos[i][3] + 1
        
        # Define the y and x ranges for the slice
        y_start, y_end = int(realPos[i][1]), int(realPos[i][0])
        x_start, x_end = int(realPos[i][3]), int(realPos[i][2])

        # ==========================================================
        # --- THIS IS THE FIX ---
        # 1. Extract the Region of Interest (ROI) from the u and v flow fields.
        u_roi = u[y_start:y_end, x_start:x_end]
        v_roi = v[y_start:y_end, x_start:x_end]

        # 2. Check if the resulting slice (ROI) is empty.
        if u_roi.size == 0:
            # 3. If it's empty, assign 0 to prevent a RuntimeWarning and NaN.
            data[i][0] = 0
            data[i][1] = 0
        else:
            # 4. If it's not empty, calculate the mean as normal.
            data[i][0] = u_roi.mean()
            data[i][1] = v_roi.mean()
        # ==========================================================
            
    return data

def main_test():
    # This test function can be used for isolated debugging
    try:
        weight = Weight_matrix().get_weight_matrix()
        # You'll need to ensure these test files exist for the test to run
        ab_img=cv2.imread('../ref_data/ab_fg_pics/105.png') # Note: png now
        origin=cv2.imread('../ref_data/original_pics/105.tif')
        
        if ab_img is None or origin is None:
            print("Could not load test images. Skipping main_test().")
            return
            
        u_seq_abnormal = scipy.io.loadmat('../ref_data/u_seq_abnormal.mat')['u_seq_abnormal']
        v_seq_abnormal = scipy.io.loadmat('../ref_data/v_seq_abnormal.mat')['v_seq_abnormal']

        ab_img = cv2.cvtColor(ab_img, cv2.COLOR_BGR2GRAY)
        origin= cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
        mask=cv2.bitwise_and(origin,origin,mask=ab_img)
        cv2.imshow('masked',mask)
        cv2.imshow('ab_img',ab_img)
        cv2.imshow('u_img105_original', u_seq_abnormal[:, :, 105])
        cv2.imshow('v_img105_original', v_seq_abnormal[:, :, 105])
        cv2.imshow('u_img105_after_weightMat',u_seq_abnormal[:,:,105]*weight.reshape(-1,1))
        cv2.imshow('v_img105_after_weightMat',v_seq_abnormal[:,:,105]*weight.reshape(-1,1))
        cv2.imshow('original',origin)
        key=cv2.waitKey(0)
        if key==27:
            cv2.destroyAllWindows()
    except Exception as e:
        print(f"An error occurred in main_test: {e}")

if __name__ == "__main__":
    # The main_test function will only run if you execute this script directly.
    # It will not run when imported by core.py.
    # main_test()
    print("getFeatureUV.py loaded as a module.")