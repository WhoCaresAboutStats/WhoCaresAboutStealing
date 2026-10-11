from ultralytics import YOLO
import os

model = YOLO("yolo11s-pose.pt")
INPUT = r"C:\Users\nitro\WhoCaresAboutStealDecision\runs\detect\predict-3\Lead.jpg"
STEM = os.path.basename(INPUT)
OUTPUT = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\IN\Test\PoseBaseConnect"

results = model.track(source=INPUT, save=True, project=OUTPUT, exist_ok=True)
