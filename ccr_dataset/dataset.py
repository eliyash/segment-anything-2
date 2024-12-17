import os
import random
from pathlib import Path

import cv2
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
from torchvision import transforms


class BaseDataset(Dataset):

    def __init__(self, annotation_level, config, train=True, transform=None):
        self.annotation_level = annotation_level
        self.train = train
        self.img_dir = os.path.join(config['dataset_dir'], 'frames')
        self.img_fmt = config['img_fmt']
        self.metadata_dir = config['metadata_dir']

        self._load_metadata()
        #self._check_integrity()

    def _load_metadata(self):
        classes = pd.read_csv(os.path.join(self.metadata_dir, 'lists', 'classes.txt'), sep=',', header=0)
        # Remove negative class for frame-level
        if self.annotation_level == 'frame':
            classes = classes[classes['name'] != 'NEGATIVE']

        self.class_map = {k: v for k, v in zip(classes['name'], classes['idx'])}
        train_test_split = pd.read_csv(os.path.join(self.metadata_dir, 'lists', 'videos.txt'), sep=',', header=0)
        data = pd.read_csv(os.path.join(self.metadata_dir, 'annotations', '%s_data.csv' % self.annotation_level),
                           sep=',', header=0)
        data = data[data['label'].isin(list(self.class_map.keys()))]
        # pdb.set_trace()
        if self.train:
            target_videos = train_test_split['video'][train_test_split['set'] == 'train']
        else:
            target_videos = train_test_split['video'][train_test_split['set'] == 'test']

        self.data = data[data['video'].isin(target_videos.values)]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        rand_idxs = random.sample(range(0, len(self.data)), 100)
        for idx in rand_idxs:
            sample = self.data.iloc[idx]
            filepath = os.path.join(self.img_dir, str(sample.year), sample.video, self.img_fmt % sample.frame)
            if not os.path.isfile(filepath):
                print(filepath)
                return False

        return True

    def __len__(self):
        return len(self.data)


class DetectionDataset(BaseDataset):

    def __init__(self, annotation_level, config, train=True, transform=None):
        super(DetectionDataset, self).__init__(annotation_level, config, train=train, transform=transform)
        self.crop_margin = config[annotation_level]['crop_margin']

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.img_dir, str(sample.year), sample.video, self.img_fmt % sample.frame)
        target = self.class_map[sample.label]
        img = Image.open(path)
        img = self._crop(img, sample.x, sample.y, sample.w, sample.h)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def _crop(self, img, x, y, w, h):
        img_width, img_height = img.size
        y = max(0, y * (1 - self.crop_margin / 2)) * img_height
        x = max(0, x * (1 - self.crop_margin / 2)) * img_width
        h = (min(y + h * (1 + self.crop_margin / 2), img_height) - y) * img_height
        w = min(x + w * (1 + self.crop_margin / 2), img_width) - x
        img = transforms.functional.crop(img, y, x, h, w)

        return img

    def visualise(self, idx):
        sample = self.data.iloc[idx]
        frame_index = sample.frame

        path = os.path.join(self.img_dir, str(sample.year), sample.video, self.img_fmt % frame_index)
        frame = Image.open(path)

        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        img_width, img_height = img.size
        # scaling fix, as annotations were scaled by width in both axes
        scaling_horizontal_val = img_width / img_height

        # find all frame annotations
        prev_iterator = 1
        while self.data.iloc[idx-prev_iterator].frame == frame_index:
            prev_iterator += 1

        next_iterator = 1
        while self.data.iloc[idx+next_iterator].frame == frame_index:
            next_iterator += 1

        for index in range(idx-prev_iterator, idx+next_iterator+1):
            sample = self.data.iloc[index]
            x, y, w, h = sample.x, sample.y * scaling_horizontal_val, sample.w, sample.h
            draw.rectangle([(x * img_width, y*img_height), ((x+w)*img_width, (y+h)*img_height)], width=1)
            draw.text((x * img_width, y*img_height), sample.label)

        return img

    def get_frame_from_video_with_annotations(self, idx):
        root = Path(self.img_dir).parent / 'videos'

        sample = self.data.iloc[idx]
        video_name = sample.video
        video_year = sample.year
        frame_index = sample.frame
        video_path = root / str(video_year) / video_name
        cap = cv2.VideoCapture(video_path.as_posix())
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        img_height, img_width, _ = frame.shape
        scaling_horizontal_val = img_width / img_height

        # find all frame annotations
        prev_iterator = 1
        while self.data.iloc[idx-prev_iterator].frame == frame_index:
            prev_iterator += 1

        next_iterator = 1
        while self.data.iloc[idx+next_iterator].frame == frame_index:
            next_iterator += 1

        bboxes = {}
        for index in range(idx-prev_iterator, idx+next_iterator+1):
            sample = self.data.iloc[index]
            x, y, w, h = sample.x, sample.y * scaling_horizontal_val, sample.w, sample.h
            bboxes[sample.label] = (x, y, w, h)

        return frame, (video_name, video_year, frame_index), bboxes


class FrameDataset(BaseDataset):

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.img_dir, str(sample.year), sample.video, self.img_fmt % sample.frame)
        labels = sample.label.split(' ')
        targets = [self.class_map[l] for l in labels]
        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, targets


dataset_dict = {'frame': FrameDataset,
                'face': DetectionDataset,
                'body': DetectionDataset
                }

def get_dataset(anno_level, config, train=True, transform=None):

    if anno_level not in dataset_dict:
        print('Please choose annotation level from one of: ', dataset_dict.keys())
        raise NotImplementedError
    else:
        dataset = dataset_dict[anno_level](anno_level, config, train=train, transform=transform)

    return dataset