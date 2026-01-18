import os
import cv2
import csv
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = './sample.mp4'
MODEL_PATH = './best.pt'
THRESHOLD = 0.75
CROP_DIR = './crops'
CSV_OUT = './staff_logs.csv'
VIDEO_OUT = './staff_tracked_out.mp4'

# Create folder for images if it doesn't exist
if not os.path.exists(CROP_DIR):
    os.makedirs(CROP_DIR)

# 1. Load the trained YOLO model
model = YOLO(MODEL_PATH)

# 2. Setup Video Input and Output
cap = cv2.VideoCapture(VIDEO_PATH)
W, H = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Use 'mp4v' codec for standard MP4 compatibility
out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

# 3. Setup CSV Logging
csv_file = open(CSV_OUT, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['frame', 'ID', 'x', 'y'])

# Tracking variables
saved_ids = set() # To save only one picture per unique person
frame_count = 0

print("Processing video... Please wait.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run tracking on the current frame
    # persist=True ensures IDs stay the same across frames
    results = model.track(frame, persist=True, verbose=False)[0]

    # Check if any staff members were detected
    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.int().cpu().tolist()
        track_ids = results.boxes.id.int().cpu().tolist()
        scores = results.boxes.conf.cpu().tolist()

        for box, track_id, score in zip(boxes, track_ids, scores):
            if score > THRESHOLD:
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                # --- DRAWING (Done before cropping to include in the JPG) ---
                label = f"ID: {track_id} Staff {cx}, {cy}"
                # Green bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Green text label above the box
                cv2.putText(frame, label, (x1, y1 - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # --- CROPPING & SAVING ---
                if track_id not in saved_ids:
                    # Add padding to capture the label and surrounding area
                    # We go 60 pixels UP, and 20 pixels out in other directions
                    pad_top = max(0, y1 - 60)
                    pad_bottom = min(H, y2 + 20)
                    pad_left = max(0, x1 - 20)
                    pad_right = min(W, x2 + 20)
                    
                    # Define the crop from the ALREADY DRAWN frame
                    staff_crop_img = frame[pad_top:pad_bottom, pad_left:pad_right]
                    
                    if staff_crop_img.size > 0:
                        img_path = os.path.join(CROP_DIR, f"staff_{track_id}.jpg")
                        cv2.imwrite(img_path, staff_crop_img)
                        saved_ids.add(track_id)

                # --- LOGGING ---
                csv_writer.writerow([frame_count, track_id, cx, cy])

    # Write the annotated frame to the output video
    out.write(frame)
    frame_count += 1

# Cleanup resources
cap.release()
out.release()
csv_file.close()

print(f"Finished! Processed Video: {VIDEO_OUT}, Data Log: {CSV_OUT}, Images saved in: {CROP_DIR}")