import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from arcface.utils.utils_config import get_config
from backbones import get_model


@torch.no_grad()
def inference(weight, cfg, imgs_folder, output_folder, image_size=112):
    net = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    net.load_state_dict(torch.load(weight))
    net.eval()

    # transform = transforms.Compose([
    #         transforms.ToPILImage(),
    #         transforms.Resize((image_size, image_size)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #      ])
    for dataset_type_folder in imgs_folder.iterdir():
        classes = []
        features = []
        for index, individual_folder in enumerate(sorted(dataset_type_folder.iterdir())):
            for img_path in list(individual_folder.iterdir()):
                img = cv2.imread(img_path.as_posix())
                img = cv2.resize(img, (112, 112))

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.transpose(img, (2, 0, 1))
                img = torch.from_numpy(img).unsqueeze(0).float().cuda()
                img.div_(255).sub_(0.5).div_(0.5)
                feat = net(img).cpu().numpy()

                classes.append(index)
                features.append(feat[0])

        output_folder_by_type = output_folder / dataset_type_folder.name
        output_folder_by_type.mkdir(parents=True, exist_ok=True)

        np.save(output_folder_by_type / f'embeddings.npy', np.array(features))
        np.save(output_folder_by_type / 'labels.npy', np.array(classes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--config', type=str, default="configs/chimp.py")
    parser.add_argument('--training_folder', type=str, default='D:/training_output/arcface/')
    parser.add_argument('--training_name', type=str, default='2025_01_24__00_15_17')
    parser.add_argument('--output_folder', type=str, default='D:/inference/arcface/')
    parser.add_argument('--imgs_folder', type=str, default='D:/faces_dataset')

    args = parser.parse_args()

    weight_path = Path(args.training_folder) / args.training_name / 'model.pt'
    output_folder = Path(args.output_folder) / args.training_name
    inference(weight_path, get_config(args.config), Path(args.imgs_folder), output_folder)
