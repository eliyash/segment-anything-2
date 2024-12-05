import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_boxes, xyxy2xywh)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


@torch.no_grad()
def run(
        source,
        output_path,
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

    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    # Use OpenCV to load images
    for img_path in Path(source).glob('*'):
        print(time.strftime('%Y-%m-%d %H:%M:%S'), img_path)
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            im0s = cv2.imread(str(img_path))
            assert im0s is not None, f"Image Not Found {img_path}"
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

            for i, det in enumerate(pred):  # per image
                p = img_path
                annotator = Annotator(im0s, line_width=line_thickness, example=str(names))
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0s.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                im0s = annotator.result()
                cv2.imwrite(str(output_path / f'{p.stem}_detected{p.suffix}'), im0s)


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
    parser.add_argument('--output_path', )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()

    opt.weights = r"C:\Users\Eliahu\Downloads\coco_datasets\chimps\end_of_train\weights\best.pt"
    opt.output_path = r"C:\Workspace\ChimpanzeesThesis\segment-anything-2\yolov9-main\runs\mytest_frames_collection_end_of_train"
    opt.source = r"D:\frames_collection"
    run(**vars(opt))
