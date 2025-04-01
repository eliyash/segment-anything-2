import cv2
import numpy as np

LK_PARAMS = dict(
    winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

def _create_trajectory_frame(inliers, frame, valid_new, valid_prev):
    trajectory_frame = frame.copy()
    # If no inliers mask was returned, assume all points are inliers.
    if inliers is None:
        inliers = np.ones((len(valid_prev), 1), dtype=bool)
    inliers = inliers.flatten()
    # Draw the movement trajectories:
    # - Green: points that agree with the global (camera) motion.
    # - Red: points that do not agree (likely local object motion).
    for (pt_prev, pt_new, inlier) in zip(valid_prev, valid_new, inliers):
        prev_pt = tuple(np.int32(pt_prev.ravel()))
        new_pt = tuple(np.int32(pt_new.ravel()))
        # Draw previous point in orange.

        prev_point_color = (0, 165, 255)
        new_point_color = (0, 255, 0) if inlier else (0, 0, 255)
        line_color = (0, 255, 255)
        cv2.circle(trajectory_frame, prev_pt, 3, prev_point_color, -1)
        # Draw new point in green.
        cv2.circle(trajectory_frame, new_pt, 3, new_point_color, -1)
        # Draw line from previous to new point in yellow.
        cv2.line(trajectory_frame, prev_pt, new_pt, line_color, 2)
    return trajectory_frame


def calc_optical_flow(frame, prev_frame, prev_bboxes):
    """
    Processes the current frame to:
      - Compute the optical flow.
      - Estimate a global affine transform (capturing camera movement and zoom)
        based on the majority of feature points (inliers).
      - Identify points that do not agree with the global motion (likely moving objects).
      - Return:
            new_tracking_data: updated points to track.
            transformed_prev_frame: the previous frame warped using the estimated transform.
            trajectory_frame: visualization of the point trajectories
                              (green for global motion/inliers, red for local motion/outliers).
    """
    # Convert current frame to grayscale.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the previous frame to grayscale.
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # If no previous tracking data exists, detect features in the previous frame.
    camera_tracking_points = cv2.goodFeaturesToTrack(prev_frame_gray, maxCorners=2000, qualityLevel=0.01, minDistance=50, blockSize=7)
    camera_tracking_points_tags = [-1]*len(camera_tracking_points)

    bboxes_tracking_points = []
    bboxes_tracking_points_tags = []
    for ind, (_, (x_start, x_end), (y_start, y_end)) in enumerate(prev_bboxes):
        bbox_mask = np.zeros_like(prev_frame_gray, dtype=np.uint8)
        bbox_mask[y_start:y_end, x_start:x_end] = 255
        bbox_tracking_points = cv2.goodFeaturesToTrack(
            prev_frame_gray, maxCorners=100, qualityLevel=0.001, minDistance=4, blockSize=7, mask=bbox_mask
        )
        if bbox_tracking_points is not None:
            bboxes_tracking_points.append(bbox_tracking_points)
            bboxes_tracking_points_tags.append([ind]*len(bbox_tracking_points))


    # Compute optical flow to get new positions of the tracked feature points.
    all_tracking_points = np.concatenate((camera_tracking_points, *bboxes_tracking_points), axis=0)
    all_tracking_points_indexes = np.concatenate((camera_tracking_points_tags, *bboxes_tracking_points_tags), axis=0)
    new_tracking_data, status, err = cv2.calcOpticalFlowPyrLK(
        prev_frame_gray, frame_gray, all_tracking_points, None, **LK_PARAMS
    )

    # If optical flow fails, return defaults.
    if new_tracking_data is None or status is None:
        return None, [], frame.copy()

    # Select only the valid points where tracking was successful.
    status = status.flatten()
    valid_prev = all_tracking_points[status == 1]
    valid_new = new_tracking_data[status == 1]
    valid_tracking_points_indexes = all_tracking_points_indexes[status == 1]

    camera_valid_prev = valid_prev[valid_tracking_points_indexes == -1]
    camera_valid_new = valid_new[valid_tracking_points_indexes == -1]

    # Estimate an affine transform that represents the global camera movement.
    # This transformation captures translation, rotation, and scaling.
    global_camera_transform, inliers = cv2.estimateAffinePartial2D(camera_valid_prev, camera_valid_new, method=cv2.RANSAC)

    per_bbox_transform = []
    for ind in range(len(prev_bboxes)):
        valid_indexes = (valid_tracking_points_indexes == ind)
        if np.any(valid_indexes):
            bbox_valid_prev = valid_prev[valid_indexes]
            bbox_valid_new = valid_new[valid_indexes]
            bbox_camera_transform, _ = cv2.estimateAffinePartial2D(bbox_valid_prev, bbox_valid_new, method=cv2.RANSAC)

        else:
            bbox_camera_transform = None
        per_bbox_transform.append(bbox_camera_transform)

    # Prepare the trajectory visualization frame (overlay drawn on the current frame).
    trajectory_frame = _create_trajectory_frame(inliers, frame, valid_new, valid_prev)

    # Return the updated tracking data along with the two visualizations.
    return global_camera_transform, per_bbox_transform, trajectory_frame
