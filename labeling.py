import numpy as np
import cv2

def labeling(pos, abnormal_fg_img, gate=0.5):
    """
    Labels bounding boxes as normal (0) or abnormal (1).
    This version includes a robust thresholding step to ensure masks are
    interpreted correctly as pure black and white.
    """
    if abnormal_fg_img is None:
        # If no mask file exists for a frame, all labels are 0.
        return pos, np.zeros(pos.shape[0], dtype=int)

    # --- THIS IS THE FINAL, DEFINITIVE FIX ---
    # 1. Convert to grayscale to handle any format (like RGBA).
    if len(abnormal_fg_img.shape) > 2:
        abnormal_fg_img = cv2.cvtColor(abnormal_fg_img, cv2.COLOR_BGR2GRAY)

    # 2. Apply a binary threshold. This is the crucial step.
    # It converts the mask into a perfect black-and-white image.
    # Any pixel value above 10 is set to 255 (white).
    # Everything else (including the 5.0 we saw) is set to 0 (black).
    _ , thresh_mask = cv2.threshold(abnormal_fg_img, 10, 255, cv2.THRESH_BINARY)
    # --- END FIX ---

    labels = []
    # Now, we use the clean, thresholded mask for all calculations.
    for thePos in pos:
        y_start, y_end = int(thePos[1]), int(thePos[0])
        x_start, x_end = int(thePos[3]), int(thePos[2])
        
        # Get the region of interest from the CLEAN mask
        roi = thresh_mask[y_start:y_end, x_start:x_end]
        
        # The 'gate' is now redundant because our mask is either 0 or 255,
        # but we keep the logic. If any white pixel exists, the mean will be > 0.5.
        if roi.size > 0 and roi.mean() > gate:
            labels.append(1)  # Abnormal
        else:
            labels.append(0)  # Normal

    return pos, np.array(labels)