import argparse
import hashlib
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock

import cv2
import numpy as np
import torch

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from yolov9_main.kalman_filter import track_objects, apply_transform_to_tracked_objects, _predict_center
from yolov9_main.match_bbox_in_video import transform_bbox, transform_all_bboxes
from yolov9_main.optical_flow import calc_optical_flow, init_optical_flow_on_frame


def inference_image(
        input_image,
        model,
        image_size=640,  # inference size (pixels)
        conf_thres=0.50,  # confidence threshold
        iou_thres=0.45,  # NMS IoU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
):
    stride, names, pt = model.stride, model.names, model.pt
    # Resize and pad image
    im = letterbox(input_image, image_size, stride=stride, auto=pt)[0]
    # Convert to RGB, to 3xHxW, float
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im, augment=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    return get_reshaped_bboxes(input_image.shape, im.shape[2:], pred)


def get_reshaped_bboxes(orig_shape, pred_shape, pred):
    boxes = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(pred_shape, det[:, :4], orig_shape).round()
            for *xyxy, conf, cls in reversed(det):
                xyxy = list(map(int, xyxy))

                box = int(cls), (xyxy[1], xyxy[3]), (xyxy[0], xyxy[2])
                boxes.append(box)
    return boxes


def uid_to_color(uid):
    # Convert UID to string and hash it
    uid_str = str(uid).encode('utf-8')
    hash_digest = hashlib.md5(uid_str).digest()

    # Use the first three bytes as RGB
    r, g, b = hash_digest[0], hash_digest[1], hash_digest[2]
    return r, g, b


def draw_bboxs_on_frame(frame, tracked_objects):
    new_frame = frame.copy()
    for uid, data in tracked_objects.items():
        (cls, (xs, xe), (ys, ye)) = data['last_bbox']
        color = uid_to_color(uid)
        cv2.rectangle(new_frame, (ys, xs), (ye, xe), color, 2)
        l_thickness = 1 if data['missing_frames'] > 0 else 2
        text = f'missing_{uid}' if data['missing_frames'] > 0 else uid
        cv2.putText(new_frame, text, (ys, xs + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, l_thickness)

        x, y = _predict_center(data["kalman"]).astype(int)
        cv2.circle(new_frame, (y, x), 2, color, 2)
    return new_frame


def get_args_parser():
    parser = argparse.ArgumentParser("YOLOX Eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="local rank for dist training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--test",
        dest="test",
        default=False,
        action="store_true",
        help="Evaluating on test-dev set.",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument("--min-box-area", type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def simple_movement_capture(cap):
    for _ in range(10):
        cap.read()
    ret, prev_frame = cap.read()
    orig_frame = prev_frame

    i = 1
    move_in_x = True

    def simple_cap_read():
        nonlocal i
        frame = np.zeros_like(orig_frame)
        if move_in_x:
            frame[:, i:] = orig_frame[:, :-i]
        else:
            frame[i:, :] = orig_frame[:-i, :]

        i += 20
        return True, frame

    cap = Mock()
    cap.read = simple_cap_read
    return cap

def draw_byte_track(frame, online_targets):
    frame = frame.copy()
    for t in online_targets:
        tlwh = t.tlwh  # top-left width-height
        tid = t.track_id
        x1, y1, w, h = tlwh
        print(x1, y1, w, h)
        cv2.rectangle(frame, (int(y1), int(x1)), (int(y1 + h), int(x1 + w)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {tid}', (int(y1), int(x1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
                    2)
    return frame

@torch.no_grad()
def run(
        source,
        weights,  # model.pt path(s)
        image_size=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IoU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=None, fp16=half)

    def inference_image_by_frame(in_frame):
        return inference_image(
            in_frame, model, image_size, conf_thres, iou_thres, max_det, device, classes, agnostic_nms
        )

    video_paths = [file_path for file_path in Path(source).glob('*') if file_path.suffix.lower() in ['.mp4', '.avi', '.mov']]

    class LoggingMock(Mock):
        def __getattr__(self, name):
            print(f"Accessed property: {name}")
            return super().__getattr__(name)

    args = LoggingMock()

    args.track_thresh=0.5
    args.track_buffer=30
    args.match_thresh=0.7
    args.mot20=False
    tracker = BYTETracker(args)

    for video_path in video_paths:
        print(time.strftime('%Y-%m-%d %H:%M:%S'), video_path)
        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"Video Not Found {video_path}"

        # cap = simple_movement_capture(cap)

        tracked_objects = {}
        ret, prev_frame = cap.read()
        # optical_flow_state = init_optical_flow_on_frame(prev_frame)
        # full_motion_transform = np.eye(3)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # optical_flow_state, motion_transform, _ = calc_optical_flow(frame, prev_frame, optical_flow_state)
                # full_motion_transform = full_motion_transform @ motion_transform
                detected_bboxes = inference_image_by_frame(frame)
                # transformed_detected_bboxes = transform_all_bboxes(detected_bboxes, full_motion_transform)

                dets_for_tracker = np.array([(x1, y1, x2, y2, 0.8) for cls, (x1, x2), (y1, y2) in detected_bboxes], dtype=float)
                if len(dets_for_tracker.shape) == 1:
                    dets_for_tracker = np.zeros((0, 5), dtype=float)
                # print(dets_for_tracker.shape)
                online_targets = tracker.update(dets_for_tracker)
                print('dets_for_tracker', dets_for_tracker, 'online_targets', online_targets, detected_bboxes)
                #
                # if motion_transform is not None:
                #     apply_transform_to_tracked_objects(tracked_objects, motion_transform)
                #
                # tracked_objects = track_objects(detected_bboxes, tracked_objects)

                cv2.imshow('frame', draw_byte_track(frame, online_targets))
                # cv2.imshow('frame', draw_bboxs_on_frame(frame, tracked_objects))
                cv2.waitKey(1)
            else:
                break
        cap.release()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, help='model path(s)')
    parser.add_argument('--source', type=str, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--image_size', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max_det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic_nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_opt()

    opt.weights = r"C:\Users\Eliahu\Downloads\best_4d8126c8128a4603bfd69daa922e8d38.pt"
    opt.source = r"D:\per_signal_videos_fixed_interlacing"
    opt.conf_thres = 0.5
    run(**vars(opt))


if __name__ == "__main__":
    main()