from ultralytics import YOLO
import cv2
import os

BASE_MODEL = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\OUT\v4\weights\best.pt"
POSE_MODEL = r"C:\Users\nitro\WhoCaresAboutStealDecision\Torch\Main\yolo11s-pose.pt"

base_model = YOLO(BASE_MODEL)
pose_model = YOLO(POSE_MODEL)

TEST_IMAGE = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\IN\Test\Images\Lead.png"
STEM = os.path.basename(TEST_IMAGE)

frame = cv2.imread(TEST_IMAGE)

base_results = base_model(frame)
pose_results = pose_model(frame)

base_boxes = base_results[0].boxes.xyxy

if len(base_boxes) == 0:
    print("No base detected")
    exit()

# Use the first detected base
x1, y1, x2, y2 = base_boxes[0]
base_center = (int((x1+x2)/2), int((y1+y2)/2))

keypoints = pose_results[0].keypoints.xy

if len(keypoints) == 0:
    print("No runner detected")
    exit()

runner_kpts = keypoints[0]
left_ankle = runner_kpts[15]
left_ankle = (int(left_ankle[0]), int(left_ankle[1]))

cv2.circle(frame, left_ankle, 6, (0, 0, 255), -1)
cv2.circle(frame, base_center, 6, (0, 255, 0), -1)

cv2.line(frame, left_ankle, base_center, (255, 255, 0), 3)

OUTPUT_FOLDER = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\IN\Test\PoseBaseConnect\Full"
FULL = os.path.join(OUTPUT_FOLDER, STEM)
cv2.imwrite(FULL, frame)
