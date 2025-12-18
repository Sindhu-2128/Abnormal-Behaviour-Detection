import cv2
import os

def generate_fg_pics(src_dir, dst_dir):
    """
    Reads all TIFF images from a source directory, converts them to grayscale,
    and saves them to a destination directory.
    """
    os.makedirs(dst_dir, exist_ok=True)
    print("[INFO] Generating FG pics from original_pics...")

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

        # Convert to grayscale for the foreground
        fg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save the new foreground image with the same name
        save_path = os.path.join(dst_dir, img_name)
        cv2.imwrite(save_path, fg)
        print(f"[OK] Saved FG image: {save_path}")

    print(f"\n[DONE] Successfully generated {len(image_files)} FG pics in: {dst_dir}")

# --- Main execution block ---
if __name__ == "__main__":
    # Define source and destination paths using raw strings
    source_directory = r'../ref_data/normal/original_pics'
    destination_directory = r'../ref_data/normal/fg_pics'
    
    generate_fg_pics(source_directory, destination_directory)