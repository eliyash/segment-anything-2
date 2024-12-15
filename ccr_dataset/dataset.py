import os
import random
# import pdb
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
        path = os.path.join(self.img_dir, str(sample.year), sample.video, self.img_fmt % sample.frame)
        img = Image.open(path)
        img_width, img_height = img.size
        draw = ImageDraw.Draw(img)
        draw.rectangle([(sample.x * img_width, sample.y*img_height), ((sample.x+sample.w)*img_width, (sample.y+sample.h)*img_height)], width=1)
        draw.text((sample.x, sample.y), sample.label)
        iterator=1
        while True:
            sample_tmp = self.data.iloc[idx-1]
            if sample_tmp.frame != sample.frame:
                break
            target = self.class_map[sample.label]
            draw.rectangle([(sample_tmp.x * img_width, sample_tmp.y * img_height),
                            ((sample_tmp.x + sample_tmp.w) * img_width, (sample_tmp.y + sample_tmp.h) * img_height)], width=1)
            draw.text((sample_tmp.x, sample_tmp.y), sample_tmp.label)
            iterator += 1
        iterator = 1
        while True:
            sample_tmp = self.data.iloc[idx + 1]
            if sample_tmp.frame != sample.frame:
                break
            target = self.class_map[sample.label]
            draw.rectangle([(sample_tmp.x * img_width, sample_tmp.y * img_height),
                            ((sample_tmp.x + sample_tmp.w) * img_width, (sample_tmp.y + sample_tmp.h) * img_height)],
                           width=1)
            draw.text((sample_tmp.x, sample_tmp.y), sample_tmp.label)
            iterator += 1

        return img


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