import numpy as np
import os
import cv2 # Make sure cv2 is imported

def generate_pics_with_hough_lines(src_dir, dst_dir):
    """
    Reads all TIFF images from a source directory, detects lines using Hough
    Transform, and saves the original images with the lines drawn on them.
    """
    os.makedirs(dst_dir, exist_ok=True)
    print("[INFO] Generating images with overlaid Hough Lines...")

    try:
        image_files = [f for f in os.listdir(src_dir) if f.lower().endswith('.tif')]
        image_files.sort()
    except FileNotFoundError:
        print(f"[ERROR] Source directory not found: {src_dir}")
        return

    if not image_files:
        print(f"[ERROR] No .tif images found in '{src_dir}'")
        return

    print(f"[INFO] Found {len(image_files)} images to process.")

    for img_name in image_files:
        img_path = os.path.join(src_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARNING] Could not read {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 105, 130)

        # ==========================================================
        # === THIS IS THE LINE THAT HAS BEEN CHANGED             ===
        # === The threshold is now 200 instead of 89.            ===
        # === This makes the line detector much less sensitive.  ===
        # ==========================================================
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is not None:
            for line in lines[:, 0, :]:
                rho, theta = line
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(img, pt1, pt2, (0, 0, 255), 2)
        
        save_name = os.path.splitext(img_name)[0] + '.png'
        save_path = os.path.join(dst_dir, save_name)
        cv2.imwrite(save_path, img)
        print(f"[OK] Saved image with Hough Lines: {save_path}")

    print(f"\n[DONE] Successfully generated {len(image_files)} images with Hough Lines in: {dst_dir}")

# --- Main execution block (if you run this file directly) ---
if __name__ == "__main__":
    source_directory = r'S:\Abnormal\Abnormal\ref_data\normal\original_pics' # Use your actual path
    destination_directory = r'S:\Abnormal\Abnormal\ref_data\normal\pics_with_hough_lines' # Use your actual path
    
    generate_pics_with_hough_lines(source_directory, destination_directory)