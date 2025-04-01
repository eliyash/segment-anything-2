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
from yolov9_main.kalman_filter import track_objects, _predict_center, \
    _apply_inverse_transform_to_kalman_state
from yolov9_main.match_bbox_in_video import transform_bbox
from yolov9_main.optical_flow import calc_optical_flow


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

    video_paths = sorted(
        [
             file_path for file_path in Path(source).glob('*')
             if file_path.suffix.lower() in ['.mp4', '.avi', '.mov']
        ]
    )

    for video_path in video_paths:
        print(time.strftime('%Y-%m-%d %H:%M:%S'), video_path)
        cap = cv2.VideoCapture(str(video_path))
        assert cap.isOpened(), f"Video Not Found {video_path}"

        tracked_objects = {}
        ret, prev_frame = cap.read()
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                detected_bboxes = inference_image_by_frame(frame)

                uids = list(tracked_objects.keys())
                prev_bboxes = [tracked_objects[uid]["last_bbox"] for uid in uids]
                global_affine_transform, per_bbox_affine_transforms, trajectory_frame = calc_optical_flow(frame, prev_frame, prev_bboxes)

                if global_affine_transform is not None:
                    for uid, bbox_transform in zip(uids, per_bbox_affine_transforms):
                        transform = bbox_transform if bbox_transform is not None else global_affine_transform
                        tracked_objects[uid]["last_bbox"] = transform_bbox(tracked_objects[uid]["last_bbox"], transform)
                        tracked_objects[uid]["kalman"].statePost = _apply_inverse_transform_to_kalman_state(tracked_objects[uid]["kalman"].statePost, transform)
                else:
                    print('\tlost tracking')


                tracked_objects = track_objects(detected_bboxes, tracked_objects, use_kalman=True)

                prev_frame = frame
                # cv2.imshow('frame', draw_bboxs_on_frame(frame, tracked_objects))
                cv2.imshow('trajectory_frame', draw_bboxs_on_frame(trajectory_frame, tracked_objects))
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
    data_name = 'per_signal_videos_fixed_interlacing'
    data_folder = Path('D:/')
    local_data_folder = Path('C:\Workspace\ChimpanzeesThesis\data_local_copy')
    if (local_data_folder / data_name).exists():
        source = local_data_folder / data_name
    else:
        source = data_folder / data_name

    opt = parse_opt()


    opt.weights = r"C:\Users\Eliahu\Downloads\best_4d8126c8128a4603bfd69daa922e8d38.pt"
    opt.source = source.as_posix()
    opt.conf_thres = 0.5
    run(**vars(opt))


if __name__ == "__main__":
    main()