import cv2
import numpy as np
from enum import Enum, auto
import uuid


class HistoryStatus(Enum):
    NEW = auto()
    OLD = auto()
    MISSING = auto()


uid_count = 0
def generate_uid():
    global uid_count
    new_uid = uid_count
    uid_count += 1
    """Generate a new unique identifier."""
    return str(new_uid)


def real_generate_uid():
    """Generate a new unique identifier."""
    return str(uuid.uuid4())


def bbox_to_array(bbox):
    """
    Convert a bounding box tuple (cls, (x_start, x_end), (y_start, y_end))
    to a NumPy array: [x_start, y_start, x_end, y_end].
    """
    return np.array([bbox[1][0], bbox[2][0], bbox[1][1], bbox[2][1]], dtype=np.float32)


def compute_intersection_area(new_coords, prev_coords):
    """
    Compute the intersection area between new and previous bounding boxes.
    new_coords: shape (N, 4)
    prev_coords: shape (M, 4)

    Returns:
      intersection: array of shape (N, M) with intersection areas.
    """
    # Compute all intersections in a vectorized manner
    inter_xmin = np.maximum(new_coords[:, None, 0], prev_coords[None, :, 0])
    inter_ymin = np.maximum(new_coords[:, None, 1], prev_coords[None, :, 1])
    inter_xmax = np.minimum(new_coords[:, None, 2], prev_coords[None, :, 2])
    inter_ymax = np.minimum(new_coords[:, None, 3], prev_coords[None, :, 3])

    inter_w = np.maximum(0, inter_xmax - inter_xmin)
    inter_h = np.maximum(0, inter_ymax - inter_ymin)

    return inter_w * inter_h


def compute_iou_matrix(new_bboxes, prev_bboxes):
    """
    Compute the IoU matrix between new and previous bounding boxes.
    new_bboxes and prev_bboxes are lists of bbox tuples.
    """
    new_coords = np.array(list(map(bbox_to_array, new_bboxes)))  # shape (N, 4)
    prev_coords = np.array(list(map(bbox_to_array, prev_bboxes)))  # shape (M, 4)

    intersection = compute_intersection_area(new_coords, prev_coords)
    new_area = (new_coords[:, 2] - new_coords[:, 0]) * (new_coords[:, 3] - new_coords[:, 1])
    prev_area = (prev_coords[:, 2] - prev_coords[:, 0]) * (prev_coords[:, 3] - prev_coords[:, 1])
    union = new_area[:, None] + prev_area[None, :] - intersection

    return np.where(union > 0, intersection / union, 0)


def get_candidate_matches(iou_matrix, threshold):
    """
    Return candidate matches as a list of tuples (new_idx, prev_idx, iou)
    for all IoU values above the given threshold.
    """
    new_idx, prev_idx = np.where(iou_matrix > threshold)
    # Use list comprehension instead of nested for-loops
    candidates = [(i, j, iou_matrix[i, j]) for i, j in zip(new_idx, prev_idx)]
    # Sort candidates by descending IoU
    return sorted(candidates, key=lambda x: x[2], reverse=True)


def greedy_match(candidates):
    """
    Greedily assign one-to-one matches from candidate matches.

    Returns:
      matched_new: dict mapping new index to matched previous index.
      matched_prev: dict mapping previous index to matched new index.
    """
    matched_new = {}
    matched_prev = {}
    for new_idx, prev_idx, _ in candidates:
        if new_idx not in matched_new and prev_idx not in matched_prev:
            matched_new[new_idx] = prev_idx
            matched_prev[prev_idx] = new_idx
    return matched_new, matched_prev


def update_tracking(prev_bboxes_and_status, new_bboxes, iou_threshold=0.5):
    """
    Update tracked bounding boxes based on new detections.

    Parameters:
      prev_bboxes_and_status: list of previous bboxes [(cls, (x_start, x_end), (y_start, y_end)), ...]
      and list of (uid, HistoryStatus) for prev_bboxes.
                   A status of MISSING means the box was not detected in the previous frame.
      new_bboxes: list of new bboxes in the current frame.
      iou_threshold: minimum IoU required to consider a match.

    Returns:
      A combined list of (bbox, (uid, HistoryStatus)).

      - New detections with no match get a new UID and NEW status.
      - Detections with a match inherit the UID and are marked as OLD.
      - Unmatched previous bboxes are marked as MISSING unless they were already missing,
        in which case they are dropped.
    """
    if len(prev_bboxes_and_status):
        prev_bboxes, prev_status = zip(*prev_bboxes_and_status)
    else:
        prev_bboxes, prev_status = [], []
    output = []
    # Filter class labels for new and previous bboxes
    new_classes = np.array(list(map(lambda b: b[0], new_bboxes)))
    prev_classes = np.array(list(map(lambda b: b[0], prev_bboxes)))

    # Compute IoU matrix (or empty if one list is empty)
    if new_bboxes and prev_bboxes:
        iou_matrix = compute_iou_matrix(new_bboxes, prev_bboxes)
        # Zero out IoU scores for non-matching classes
        class_mask = (new_classes[:, None] == prev_classes[None, :])
        iou_matrix *= class_mask
    else:
        iou_matrix = np.zeros((len(new_bboxes), len(prev_bboxes)))

    # Get candidates with IoU above the threshold
    candidates = get_candidate_matches(iou_matrix, iou_threshold)
    matched_new, matched_prev = greedy_match(candidates)

    # Process new detections: assign OLD if matched; else, mark as NEW.
    for new_idx, bbox in enumerate(new_bboxes):
        if new_idx in matched_new:
            prev_idx = matched_new[new_idx]
            uid = prev_status[prev_idx][0]
            output.append((bbox, (uid, HistoryStatus.OLD)))
        else:
            output.append((bbox, (generate_uid(), HistoryStatus.NEW)))

    # Process previous boxes that weren't matched
    for prev_idx, bbox in enumerate(prev_bboxes):
        if prev_idx not in matched_prev:
            uid, status = prev_status[prev_idx]
            # If already missing, drop the bbox (missing for 2 consecutive frames)
            if status == HistoryStatus.MISSING:
                continue
            output.append((bbox, (uid, HistoryStatus.MISSING)))

    return output


def transform_bbox(bbox, affine_transform_prev_frame):
    """
    Transform a bounding box with an affine transformation.

    Parameters:
      bbox: tuple in the form (cls, (x_start, x_end), (y_start, y_end))
      affine_transform_prev_frame: 2x3 affine transformation matrix

    Returns:
      Transformed bbox in the same format.
    """
    cls, (x_start, x_end), (y_start, y_end) = bbox

    # Define the 4 corners of the bbox
    corners = np.array([
        [x_start, y_start],
        [x_end, y_start],
        [x_end, y_end],
        [x_start, y_end]
    ], dtype=np.float32)

    # Reshape to (N, 1, 2) as expected by cv2.transform
    corners = corners.reshape(-1, 1, 2)
    transformed_corners = cv2.transform(corners, affine_transform_prev_frame)
    transformed_corners = transformed_corners.reshape(-1, 2)

    # Compute new bbox from the transformed corners
    new_x_start = int(np.min(transformed_corners[:, 0]))
    new_x_end = int(np.max(transformed_corners[:, 0]))
    new_y_start = int(np.min(transformed_corners[:, 1]))
    new_y_end = int(np.max(transformed_corners[:, 1]))

    return (cls, (new_x_start, new_x_end), (new_y_start, new_y_end))

def transform_all_bboxes(updated_bboxes_and_status, affine_transform_prev_frame):
    # Suppose `affine_transform_prev_frame` is your 2x3 matrix and
    # `updated_bboxes_and_status` is your list of (bbox, (uid, HistoryStatus)).

    updated_transformed_bboxes_and_status = []
    for bbox, status in updated_bboxes_and_status:
        transformed_bbox = transform_bbox(bbox, affine_transform_prev_frame)
        updated_transformed_bboxes_and_status.append((transformed_bbox, status))

    return updated_transformed_bboxes_and_status

