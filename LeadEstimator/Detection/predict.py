from ultralytics import YOLO
import os

MODEL = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\OUT\v4\weights\best.pt"
INPUT = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\IN\Test\Videos\NunezTest_cut.mp4"
STEM = os.path.basename(INPUT)
OUTPUT = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\IN\Test\PoseBaseConnect"


model = YOLO(MODEL)

results = model.predict(source=INPUT, project=OUTPUT, save=True, exist_ok=True)


FULL = os.path.join(OUTPUT, STEM)

results[0].show(filename=FULL)
