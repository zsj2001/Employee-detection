import cv2
import os

# 1. Setup Directory
output_dir = "dataset/images"
os.makedirs(output_dir, exist_ok=True)

# 2. Load Video
video_path = "sample.mp4"
cap = cv2.VideoCapture(video_path)

# 3. Calculate Interval
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
max_images = 300
# Skip frames evenly to cover the whole video
step = max(1, total_frames // max_images)

print(f"Total frames available: {total_frames}")
print(f"Extracting 1 frame every {step} frames...")

count = 0
frame_idx = 0

while cap.isOpened() and count < max_images:
    success, frame = cap.read()
    if not success:
        break

    # Only save the frame if it's at the right interval
    if frame_idx % step == 0:
        count += 1
        file_path = os.path.join(output_dir, f"sample_frame_{count:03d}.jpg")
        cv2.imwrite(file_path, frame)
        
    frame_idx += 1

cap.release()
print(f"Done! Saved {count} images to '{output_dir}'.")