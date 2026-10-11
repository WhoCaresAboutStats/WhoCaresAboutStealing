import cv2
import math
import numpy as np
from ultralytics import YOLO
from sort import Sort
import os
import sys

BASE_MODEL = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\OUT\v4\weights\best.pt"
POSE_MODEL = r"C:\Users\nitro\WhoCaresAboutStealDecision\Torch\Main\yolo11s-pose.pt"

base_model = YOLO(BASE_MODEL)
pose_model = YOLO(POSE_MODEL)

INPUT = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\IN\Test\Videos\NunezTest_cut.mp4"
STEM = os.path.basename(INPUT)
FILE, EXT = os.path.splitext(STEM)
print(STEM, FILE)

OUTPUT = r"C:\Users\nitro\WhoCaresAboutStealDecision\Custom\Base\YOLO\IN\Test\PoseBaseConnect\Videos"
FILE_FULL = os.path.join(OUTPUT, FILE)
os.makedirs(FILE_FULL, exist_ok=True)

# Tracker
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

# Video input
cap = cv2.VideoCapture(INPUT)
frame_idx = 0

runner_id = None

out = None

runner_ankle_history = []

def get_left_ankle(keypoints, box):
    bx1, by1, bx2, by2 = box
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2

    best_i = None
    best_dist = float("inf")

    for i, person in enumerate(keypoints):
        px, py = person[15]  # left ankle
        dist = math.hypot(px - cx, py - cy)
        if dist < best_dist:
            best_dist = dist
            best_i = i

    if best_i is None:
        return None

    ax, ay = keypoints[best_i][15]
    return int(ax), int(ay)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    # Initialize video writer
    if out is None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(f"{FILE_FULL}\output.mp4", fourcc, 30, (w, h))

    base_results = base_model(frame)
    base_boxes = base_results[0].boxes.xyxy

    base_center = None
    if len(base_boxes) > 0:
        x1, y1, x2, y2 = base_boxes[0]
        base_center = (int((x1+x2)/2), int((y1+y2)/2))

    pose_results = pose_model(frame)
    keypoints = pose_results[0].keypoints.xy
    boxes = pose_results[0].boxes.xyxy

    # Prepare detections for SORT
    dets = []
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        dets.append([x1, y1, x2, y2, 1.0])

    dets = np.array(dets)
    tracks = tracker.update(dets)

    if frame_idx == 1:
        for track in tracks:
            tx1, ty1, tx2, ty2, tid = track
            cv2.rectangle(frame, (int(tx1), int(ty1)), (int(tx2), int(ty2)), (0,255,0), 2)
            cv2.putText(frame, f"ID {int(tid)}", (int(tx1), int(ty1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Select Runner", frame)
        cv2.waitKey(1)

        runner_id = int(input("Enter runner ID: "))

        cv2.destroyWindow("Select Runner")
        print(f"Runner ID set to {runner_id}")

        out.write(frame)
        continue

    runner_ankle = None

    for track in tracks:
        tx1, ty1, tx2, ty2, tid = track

        if int(tid) == runner_id:
            runner_ankle = get_left_ankle(keypoints, (tx1, ty1, tx2, ty2))
            break

    if runner_ankle:
        cv2.circle(frame, runner_ankle, 6, (0, 0, 255), -1)

    if base_center:
        cv2.circle(frame, base_center, 6, (0, 255, 0), -1)

    if runner_ankle and base_center:
        cv2.line(frame, runner_ankle, base_center, (255, 255, 0), 3)
        # Pixel distance between runner ankle and base
        dx = runner_ankle[0] - base_center[0]
        dy = runner_ankle[1] - base_center[1]
        pixel_dist = int(math.hypot(dx, dy))

        # Draw pixel distance
        cv2.putText(frame,
                    f"{pixel_dist}px",
                    (runner_ankle[0], runner_ankle[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

    # Write to video
    out.write(frame)

cap.release()
out.release()
print("Video saved as output.mp4")