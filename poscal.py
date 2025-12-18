# poscal.py
import cv2
import numpy as np
from skimage.measure import label, regionprops

def poscal(img):
    if img is None or img.size == 0:
        return np.zeros((0, 5)), np.zeros((240, 320), dtype=np.uint8)

    if len(img.shape) > 2:
        img = img[:, :, 0]

    # --- REFINED FILTERING ---
    # 1. Open: Removes salt-and-pepper noise from the outside.
    # 2. Close: Fills in small holes inside the blobs (e.g., gaps between legs).
    kernel = np.ones((5, 5), np.uint8)
    opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    # --- END REFINED FILTERING ---

    im_labels = label(cleaned_img)
    props = regionprops(im_labels)
    
    im_s_list = []
    for prop in props:
        if prop.area < 200: # Slightly increased minimum area
            continue
        
        min_r, min_c, max_r, max_c = prop.bbox
        box_data = [max_r, min_r, max_c, min_c, prop.area]
        im_s_list.append(box_data)
    
    if not im_s_list:
        return np.zeros((0, 5)), cleaned_img
        
    im_s = np.array(im_s_list)
    return im_s, cleaned_img