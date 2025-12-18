import cv2
import os

def video_to_tiff_frames(video_path, output_folder):
    """
    Extracts frames from a video and saves them as sequentially numbered
    TIFF images (e.g., 001.tif, 002.tif, etc.).
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: '{output_folder}'")

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return

    frame_count = 0
    print(f"Starting extraction of '{os.path.basename(video_path)}' to TIFF frames...")

    while True:
        success, frame = video_capture.read()
        if not success:
            break  # End of video

        # Create filenames like '001.tif', '002.tif', etc.
        img_name = str(frame_count + 1).zfill(3) + '.tif'
        frame_filename = os.path.join(output_folder, img_name)

        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    video_capture.release()
    print("\n--- Extraction Complete ---")
    print(f"Successfully extracted {frame_count} frames.")
    print(f"TIFF images are saved in: '{output_folder}'")

# --- Main execution block ---
if __name__ == "__main__":
    # --- CONFIGURE YOUR PATHS HERE ---
    # Use raw strings (r"...") for Windows paths to avoid errors.
    input_video_path = r"S:\Abnormal\Abnormal\ref_data\normal\video.mp4"

    # The folder where '001.tif', '002.tif', etc., will be saved.
    frames_output_folder = r"S:\Abnormal\Abnormal\ref_data\normal\original_pics"

    print(f"Preparing to save frames to '{frames_output_folder}'")
    if os.path.exists(frames_output_folder) and len(os.listdir(frames_output_folder)) > 0:
        print("Warning: Output folder is not empty. Existing files might be overwritten.")

    video_to_tiff_frames(input_video_path, frames_output_folder)