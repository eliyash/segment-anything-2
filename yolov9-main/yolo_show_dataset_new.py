from pathlib import Path

import cv2
import numpy as np

from utils.dataloaders import LoadImagesAndLabels


def main():
    # path = Path(r'D:\count_crop_and_recognise_dataset')
    path = Path(r'C:\Workspace\ChimpanzeesThesis\refactor')

    dataset_name = 'val'
    img_size = 640

    dataset = LoadImagesAndLabels(path / f'{dataset_name}.txt', img_size, 1)
    for img_torch, labels, img_path, shapes in dataset:
        (orig_h, orig_w), ((scale_w, scale_h), (crop_h, crop_w)) = shapes
        crop_w, crop_h = map(int, (crop_w, crop_h))
        img = img_torch.numpy().transpose(1, 2, 0)
        # img = img[crop_w: img_size-crop_w, crop_h: img_size-crop_h]
        labels = labels.numpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        im_w, im_h, _ = img.shape
        color = (0, 255, 0)
        for label in labels:
            _, c, x, y, w, h = label
            x, y, w, h = x*im_w, y*im_h, w*im_w, h*im_h
            cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 1)

        cv2.imshow('img', img)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
