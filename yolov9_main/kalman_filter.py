import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from yolov9_main.match_bbox_in_video import generate_uid
MAX_NUMBER_OF_MISSING_FRAMES = 8

def _enforce_float32_kalman(kf):
    kf.transitionMatrix = kf.transitionMatrix.astype(np.float32)
    kf.measurementMatrix = kf.measurementMatrix.astype(np.float32)
    kf.processNoiseCov = kf.processNoiseCov.astype(np.float32)
    kf.measurementNoiseCov = kf.measurementNoiseCov.astype(np.float32)
    kf.errorCovPost = kf.errorCovPost.astype(np.float32)
    kf.statePost = kf.statePost.astype(np.float32)
    kf.statePre = kf.statePre.astype(np.float32)

    if hasattr(kf, "gain"):
        kf.gain = kf.gain.astype(np.float32)


def _create_kalman_filter(bbox):
    kf = cv2.KalmanFilter(4, 2)  # 4 state dims: x, y, dx, dy | 2 measurements: x, y

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    kf.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(4, dtype=np.float32)

    center_x, center_y = _bbox_center(bbox)
    kf.statePost = np.array([[center_x], [center_y], [0], [0]], dtype=np.float32)  # shape: (4, 1)
    _enforce_float32_kalman(kf)

    return kf

def _predict_center(kf):
    pred = kf.predict()
    return np.array([pred[0][0], pred[1][0]])

def _correct_kalman(kf, bbox):
    box_center = _bbox_center(bbox)
    measurement = np.array([[box_center[0]], [box_center[1]]], dtype=np.float32)
    kf.correct(measurement)

def _bbox_center(bbox):
    cls, (x_start, x_end), (y_start, y_end) = bbox
    return np.array([(x_end - x_start) / 2, (y_end - y_start) / 2])

def _compute_cost_matrix(predictions, detections):
    cost = np.zeros((len(predictions), len(detections)), dtype=np.float32)
    for i, pred in enumerate(predictions):
        for j, det in enumerate(detections):
            cost[i, j] = np.linalg.norm(pred - _bbox_center(det))
    return cost

def match_predictions_to_detections(tracked_objects, detections, distance_threshold=50):
    ids = list(tracked_objects.keys())
    predictions = [_predict_center(tracked_objects[uid]["kalman"]) for uid in ids]

    cost_matrix = _compute_cost_matrix(predictions, detections)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = set(ids)
    unmatched_detections = set(range(len(detections)))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < distance_threshold:
            matches.append((ids[r], c))
            unmatched_tracks.remove(ids[r])
            unmatched_detections.remove(c)

    return matches, list(unmatched_tracks), list(unmatched_detections)

def update_tracks(tracked_objects, detections, matches, unmatched_tracks, unmatched_detections):
    updated_objects = {}

    # Update matched
    for uid, det_idx in matches:
        obj = tracked_objects[uid]
        _correct_kalman(obj["kalman"], detections[det_idx])
        obj["last_bbox"] = detections[det_idx]
        obj["missing_frames"] = 0
        updated_objects[uid] = obj

    # Handle unmatched (aging)
    for uid in unmatched_tracks:
        obj = tracked_objects[uid]
        obj["missing_frames"] += 1
        if obj["missing_frames"] <= MAX_NUMBER_OF_MISSING_FRAMES:
            updated_objects[uid] = obj  # keep aging ones

    # Add new
    for det_idx in unmatched_detections:
        new_kf = _create_kalman_filter(detections[det_idx])
        uid = generate_uid()
        updated_objects[uid] = {
            "id": uid,
            "kalman": new_kf,
            "last_bbox": detections[det_idx],
            "missing_frames": 0
        }

    return updated_objects

def track_objects(detections, tracked_objects):
    matches, unmatched_tracks, unmatched_detections = match_predictions_to_detections(
        tracked_objects, detections)

    tracked_objects = update_tracks(tracked_objects, detections, matches, unmatched_tracks, unmatched_detections)
    return tracked_objects
