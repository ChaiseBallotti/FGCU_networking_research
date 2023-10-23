from ultralytics import YOLO
model = YOLO('yolov8s.pt')
results = model(source="\visdrone_dataset\DSM\encoded\DSM1_264_1080p.mp4", show=True, save=True)