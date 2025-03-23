import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from facenet.inception_v3_train import ALL_CLASS_INDEX_TO_NAMES
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


def inference_image(
        im0s,
        model,
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IoU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
):

    stride, names, pt = model.stride, model.names, model.pt
    # Resize and pad image
    im = letterbox(im0s, imgsz, stride=stride, auto=pt)[0]
    # Convert to RGB, to 3xHxW, float
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im, augment=augment)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    return im, pred


def add_pred_to_image(im0s, im, pred):
    boxes = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
            for *xyxy, conf, cls in reversed(det):
                xyxy = list(map(int, xyxy))

                box = int(cls), (xyxy[1], xyxy[3]), (xyxy[0], xyxy[2])
                boxes.append(box)
                # face = im0s[xyxy[1]: xyxy[3], xyxy[0]: xyxy[2]].copy()

                # cv2.rectangle(im0s, xyxy[:2], xyxy[2:], (0, 255, 0), 2)
                #
                # cv2.imshow('bbox', im0s)
                # cv2.imshow('face', face)
                # cv2.waitKey()
    return boxes


@torch.no_grad()
def run(
        source,
        # output_path,
        weights,  # model.pt path(s)
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IoU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # output_path = Path(output_path)
    # output_path.mkdir(exist_ok=True, parents=True)

    # Use OpenCV to load videos
    for video_path in Path(source).glob('*'):
        print(time.strftime('%Y-%m-%d %H:%M:%S'), video_path)
        if video_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            # annotation_file = (output_path / f'{video_path.stem}.json')
            # if annotation_file.exists():
            #     print(f"Skipping {video_path.stem}")
            #     continue
            cap = cv2.VideoCapture(str(video_path))
            assert cap.isOpened(), f"Video Not Found {video_path}"

            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            while (cap.isOpened()):
                ret, im0s = cap.read()
                if ret:
                    image, pred = inference_image(
                        im0s, model, imgsz, conf_thres, iou_thres, max_det, device, classes, agnostic_nms, augment
                    )

                    boxes = add_pred_to_image(im0s, image, pred)

                    for cls, (xs, xe), (ys, ye) in boxes:
                        cv2.rectangle(im0s, (ys, xs), (ye, xe), (0, 255, 0), 2)
                        cv2.putText(im0s, ALL_CLASS_INDEX_TO_NAMES[cls], (ys, xs + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2)
                    # out.write(im0s)
                    cv2.imshow('frame', im0s)
                    cv2.waitKey(1)
                else:
                    break

            cap.release()
            # out.release()
            # annotation_file.write_text(json.dumps(video_frames_data, indent=4))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, help='model path(s)')
    parser.add_argument('--source', type=str, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # parser.add_argument('--output_path')
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()

    opt.weights = r"C:\Users\Eliahu\Downloads\best_a3124f8b12f1458db290a62b6a630179.pt"
    # opt.output_path = None
    opt.source = r"D:\per_signal_videos_fixed_interlacing"
    run(**vars(opt))
