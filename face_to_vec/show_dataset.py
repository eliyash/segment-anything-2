import cv2
import numpy as np

from face_to_vec.train import prepare_triplet_data, parse_args


def create_sample_collection_of_dataset(data_loader, axis_size=6):
    images = []
    for sample in data_loader[:axis_size ** 2]:
        image = sample[0][0].numpy().transpose(1, 2, 0)
        images.append(image)

        # cv2.imshow('image', image)
        # cv2.waitKey(0)

    images = np.array(images)
    images = images.reshape(axis_size, axis_size, *images.shape[1:])
    images = np.concatenate(images, axis=1)
    return np.concatenate(images, axis=1)


def main():
    config = parse_args()

    config.data_dir = 'D:/faces_dataset/train'
    config.batch_size = 1

    train_loader, val_loader, num_classes = prepare_triplet_data(config)

    train_images = create_sample_collection_of_dataset(train_loader, axis_size=6)
    cv2.imshow('train_images', train_images)

    val_images = create_sample_collection_of_dataset(val_loader, axis_size=6)
    cv2.imshow('val_images', val_images)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
