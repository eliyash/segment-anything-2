import argparse
import time
from pathlib import Path

import cv2
import torch
import inference_on_full_videos_updated
from facenet.inception_v3_inference import predict_classes

from models.common import DetectMultiBackend
from utils.general import check_img_size
from utils.torch_utils import select_device


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

    res = {}
    for img_path in Path(source).glob('*/*'):
        print(time.strftime('%Y-%m-%d %H:%M:%S'), img_path)
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            im0s = cv2.imread(str(img_path))
            assert im0s is not None, f"Image Not Found {img_path}"
            # Resize and pad image

            im, pred = inference_on_full_videos_updated.inference_image(
                im0s, model, imgsz, conf_thres, iou_thres, max_det, device, classes, agnostic_nms, augment
            )

            boxes = inference_on_full_videos_updated.add_pred_to_image(im0s, im, pred)
            if len(boxes) > 0:
                res[img_path] = boxes

            # if len(res) > 10:
            #     break

    images_out_folder = output_path
    images_out_folder.mkdir(exist_ok=True, parents=True)
    use_insecption_classifier = False
    for img_path, boxes in res.items():
        if len(boxes) > 0:
            image = cv2.imread(str(img_path))
            if use_insecption_classifier:
                faces = [image[xs: xe, ys: ye].copy() for _, (xs, xe), (ys, ye) in boxes]
                predicted_class_names = predict_classes(faces)
                boxes = [(cls, *bbox_data) for cls, bbox_data in zip(predicted_class_names, boxes)]

            for cls, (xs, xe), (ys, ye) in boxes:
                cv2.rectangle(image, (ys, xs), (ye, xe), (0, 255, 0), 2)
                cv2.putText(image, str(cls), (ys, xs + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.imshow('orig', image)
            # cv2.waitKey(0)
            cv2.imwrite(str(images_out_folder / img_path.name), image)


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

    opt.weights = r"C:\Users\Eliahu\Downloads\best.pt"
    # opt.output_path = r"D:\inference_frames_collection_using_ccr_train_with_orig_numbers"
    # opt.source = r"D:\frames_collection"
    opt.output_path = r"D:\inference_chimpanzee_id_data"
    opt.source = r"C:\Workspace\ChimpanzeesThesis\Chimpanzee ID Data"
    run(**vars(opt))
