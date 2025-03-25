import argparse
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from yolov9_main.match_bbox_in_video import update_tracking, HistoryStatus
from yolov9_main.optical_flow import run_optical_flow_on_frame


def inference_image(
        input_image,
        model,
        image_size=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
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


def draw_bboxs_on_frame(frame, updated_bboxes_and_status):
    new_frame = frame.copy()
    for (cls, (xs, xe), (ys, ye)), (uid, status) in updated_bboxes_and_status:
        cv2.rectangle(new_frame, (ys, xs), (ye, xe), uid_to_color(uid), 2)
        match status:
            case HistoryStatus.NEW:
                l_thickness = 3
            case HistoryStatus.OLD:
                l_thickness = 2
            case _:  # HistoryStatus.MISSING:
                l_thickness = 1

        cv2.putText(new_frame, uid, (ys, xs + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, uid_to_color(uid), l_thickness)
    return new_frame


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

    for video_path in video_paths:
        print(time.strftime('%Y-%m-%d %H:%M:%S'), video_path)
        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"Video Not Found {video_path}"

        updated_bboxes_and_status = []

        tracking_data = None
        prev_frame = None
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                tracking_data, affine_transform_prev_frame, trajectory_frame = run_optical_flow_on_frame(frame, prev_frame, tracking_data)
                detected_bboxes = inference_image_by_frame(frame)
                updated_bboxes_and_status = update_tracking(updated_bboxes_and_status, detected_bboxes, iou_threshold=0.3)

                if prev_frame is not None:
                    if affine_transform_prev_frame is not None:
                        transformed_prev_frame = cv2.warpAffine(prev_frame, affine_transform_prev_frame, (frame.shape[1], frame.shape[0]))
                        cv2.imshow('transformed_prev_frame', np.abs(frame.astype(int)-transformed_prev_frame.astype(int)).astype(np.uint8))
                    cv2.imshow('prev_frame', np.abs(frame.astype(int)-prev_frame.astype(int)).astype(np.uint8))
                    cv2.imshow('trajectory_frame', trajectory_frame)

                prev_frame = frame
                cv2.imshow('frame', draw_bboxs_on_frame(frame, updated_bboxes_and_status))
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