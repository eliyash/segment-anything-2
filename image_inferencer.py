import time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


def main():
    np.random.seed(3)
    # device = 'cuda'
    device = 'cpu'
    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    print(f'before build_sam2 {time.strftime("%Y-%m-%d %H:%M:%S")}')
    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    print(f'after build_sam2 {time.strftime("%Y-%m-%d %H:%M:%S")}')
    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    print(f'after mask_generator {time.strftime("%Y-%m-%d %H:%M:%S")}')

    image_path = 'notebooks/images/cars.jpg'

    image = np.array(Image.open(image_path).convert("RGB"))

    masks = mask_generator.generate(image)
    print(f'after generate {time.strftime("%Y-%m-%d %H:%M:%S")}')

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
