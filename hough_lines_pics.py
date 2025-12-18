import cv2
import numpy as np
import os

def generate_hough_lines_only(src_dir, dst_dir):
    """
    Reads all TIFF images from a source directory, detects lines using Hough
    Transform, and saves images of only the lines to a destination directory.
    """
    os.makedirs(dst_dir, exist_ok=True)
    print("[INFO] Generating Hough Lines images (lines only)...")

    # --- Dynamically find all image files ---
    try:
        image_files = [f for f in os.listdir(src_dir) if f.lower().endswith('.tif')]
        image_files.sort()  # Sort to ensure order (001, 002, 003...)
    except FileNotFoundError:
        print(f"[ERROR] Source directory not found: {src_dir}")
        return

    if not image_files:
        print(f"[ERROR] No .tif images found in '{src_dir}'")
        return

    print(f"[INFO] Found {len(image_files)} images to process.")

    # --- Loop through all found images ---
    for img_name in image_files:
        img_path = os.path.join(src_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARNING] Could not read {img_path}")
            continue

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 105, 130)

        # Detect lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 89)
        
        # Create a blank black image to draw the lines on
        line_img = np.zeros_like(img)

        if lines is not None:
            for line in lines[:, 0, :]:
                rho, theta = line
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                # These points extend the line to the edges of the image
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                # Draw a white line on the black image
                cv2.line(line_img, pt1, pt2, (255, 255, 255), 2)
        
        # Save the image containing only the lines as a PNG
        save_name = os.path.splitext(img_name)[0] + '.png'
        save_path = os.path.join(dst_dir, save_name)
        cv2.imwrite(save_path, line_img)
        print(f"[OK] Saved Hough Lines image: {save_path}")

    print(f"\n[DONE] Successfully generated {len(image_files)} Hough Lines images in: {dst_dir}")

# --- Main execution block ---
if __name__ == "__main__":
    # Define source and destination paths using raw strings
    source_directory = r'../ref_data/normal/original_pics'
    destination_directory = r'../ref_data/normal/hough_lines_pics'
    
    generate_hough_lines_only(source_directory, destination_directory)