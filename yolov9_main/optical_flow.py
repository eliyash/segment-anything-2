import cv2
import numpy as np


def init_optical_flow_on_frame(frame):
    # Convert current frame to grayscale.
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # If no previous frame exists, initialize a dense set of feature points.
    # Increase the number of corners for better coverage.
    points = cv2.goodFeaturesToTrack(frame_gray, maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7)
    return points


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
        pt_prev = tuple(np.int32(pt_prev.ravel()))
        pt_new = tuple(np.int32(pt_new.ravel()))
        color = (0, 255, 0) if inlier else (0, 0, 255)
        cv2.line(trajectory_frame, pt_prev, pt_new, color, 2)
        cv2.circle(trajectory_frame, pt_new, 3, color, -1)
    return trajectory_frame


def calc_optical_flow(frame, prev_frame, optical_flow_state):
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
    if optical_flow_state is None:
        optical_flow_state = cv2.goodFeaturesToTrack(prev_frame_gray, maxCorners=500, qualityLevel=0.01,
                                                     minDistance=7, blockSize=7)

    # Parameters for Lucas-Kanade optical flow.
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Compute optical flow to get new positions of the tracked feature points.
    new_tracking_data, status, err = cv2.calcOpticalFlowPyrLK(
        prev_frame_gray, frame_gray, optical_flow_state, None, **lk_params
    )

    # If optical flow fails, return defaults.
    if new_tracking_data is None or status is None:
        return optical_flow_state, prev_frame.copy(), frame.copy()

    # Select only the valid points where tracking was successful.
    status = status.flatten()
    valid_prev = optical_flow_state[status == 1]
    valid_new = new_tracking_data[status == 1]

    # Estimate an affine transform that represents the global camera movement.
    # This transformation captures translation, rotation, and scaling.
    affine_transform_prev_frame, inliers = cv2.estimateAffinePartial2D(valid_prev, valid_new, method=cv2.RANSAC)

    # Prepare the trajectory visualization frame (overlay drawn on the current frame).
    trajectory_frame = _create_trajectory_frame(inliers, frame, valid_new, valid_prev)

    # Return the updated tracking data along with the two visualizations.
    return new_tracking_data, affine_transform_prev_frame, trajectory_frame
