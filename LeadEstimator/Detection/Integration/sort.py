import numpy as np
from collections import deque

def iou(bb_test, bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
              + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return o

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.bbox = bbox
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.no_losses = 0
        self.history = deque(maxlen=10)

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.no_losses = 0
        self.history.append(bbox)

    def predict(self):
        self.no_losses += 1
        return self.bbox

class Sort:
    def __init__(self, max_age=10, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, dets):
        updated_tracks = []

        # Predict new locations
        for t in self.trackers:
            t.predict()

        unmatched_dets = list(range(len(dets)))
        unmatched_trks = list(range(len(self.trackers)))
        matches = []

        # Match detections to trackers
        for d, det in enumerate(dets):
            best_iou = 0
            best_t = None

            for t, trk in enumerate(self.trackers):
                iou_score = iou(det[:4], trk.bbox)
                if iou_score > best_iou and iou_score > self.iou_threshold:
                    best_iou = iou_score
                    best_t = t

            if best_t is not None:
                matches.append((d, best_t))

                if d in unmatched_dets:
                    unmatched_dets.remove(d)

                if best_t in unmatched_trks:   # FIXED LINE
                    unmatched_trks.remove(best_t)

        # Update matched trackers
        for d, t in matches:
            self.trackers[t].update(dets[d][:4])

        # Create new trackers for unmatched detections
        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[d][:4]))

        # Remove old trackers
        self.trackers = [t for t in self.trackers if t.no_losses <= self.max_age]

        # Return results
        for t in self.trackers:
            updated_tracks.append(np.concatenate((t.bbox, [t.id])))

        return np.array(updated_tracks)
