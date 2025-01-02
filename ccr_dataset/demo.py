import json
from pathlib import Path

import cv2
import numpy as np

from dataset import get_dataset
# import pdb

json_file = open('config.json')
config = json.load(json_file)

out_dir = Path('D:/Count Crop and Recognise dataset/frame_cv')
out_dir.mkdir(parents=True, exist_ok=True)

done_list_path = out_dir / 'done_list.json'
done_list = json.loads(done_list_path.read_text()) if done_list_path.exists() else []

for dataset_name in ['face', 'body']:  # 'frame'
    for set_type in ['train', 'test']:
        dataset = get_dataset(dataset_name, config, train=set_type == 'train')
        for i in range(0, len(dataset), 100):
            print(f'starting: {dataset_name} {set_type} {i}')
            if (dataset_name, set_type, i) in done_list:
                print(f'\tskipping')
                continue
            vis_test, (video_name, video_year, frame_index), bboxes = dataset.get_frame_from_video_with_annotations(i)
            cv_image = np.array(vis_test)
            y_size, x_size, _ = cv_image.shape
            video_folder = out_dir / str(video_year) / video_name
            video_folder.mkdir(parents=True, exist_ok=True)

            for name, bbox in bboxes.items():
                bbox = np.array(bbox)
                bbox[2:] += bbox[:2]
                bbox[::2] *= x_size
                bbox[1::2] *= y_size
                bbox = bbox.astype(int)
                cv2.rectangle(cv_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.imshow('', cv_image)
            cv2.waitKey(1)

            index_str = f'{frame_index:08d}'
            cv2.imwrite((video_folder / f'{index_str}.jpg').as_posix(), cv_image)

            (video_folder / f'{dataset_name}_{index_str}.json').write_text(json.dumps(bboxes))

            done_list.append((dataset_name, set_type, i))
            done_list_path.write_text(json.dumps(done_list))
            print(f'\tdone')


    # vis_test.save(f'C:/Workspace/ChimpanzeesThesis/Count Crop and Recognise dataset/vis/{name}.jpg')

# for j in range(0, 40, 10):
#     for i in range(0, 100, 30):
#     vis_test = dataset.visualise(1, i, j)
#     cv2.imshow('', np.array(vis_test))
#     cv2.waitKey()
#     # vis_test.save(f'C:/Workspace/ChimpanzeesThesis/Count Crop and Recognise dataset/vis/body_train_{i:05d}_{j:05d}.jpg')
#     # vis_test = BodyDataset_train.visualise(i)
#     #     vis_test.save(f'C:/Workspace/ChimpanzeesThesis/Count Crop and Recognise dataset/vis/face_test_{j}.jpg')

print('done')