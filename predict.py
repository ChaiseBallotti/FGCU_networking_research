import csv
import time
import torch
from ultralytics import YOLO

# Load YOLO model with GPU optimization
model = YOLO('yolov8s.pt').cuda()

# Open video file
video_path = r"\visdrone_dataset\DSM\encoded\DSM1_264_1080p.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Initialize variables
predictions = []
start_time = time.time()

# Loop through each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Make predictions on each frame using the YOLO model
    with torch.no_grad():
        prediction = model(frame.cuda(), size=640)
        predictions.append(prediction)

# Average the predictions from each frame
avg_prediction = sum(predictions) / len(predictions)

# Calculate the frames per second (FPS) of the video
fps = frame_count / (time.time() - start_time)

# Save the average prediction and FPS to a CSV file
with open('predictions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Average Prediction', 'FPS'])
    writer.writerow([avg_prediction, fps])
