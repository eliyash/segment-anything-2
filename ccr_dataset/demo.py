import json
from dataset import get_dataset
# import pdb

json_file = open('config.json')
config = json.load(json_file)
BodyDataset_train = get_dataset('body', config, train=True)
FrameDataset_train = get_dataset('frame', config, train=True)
FaceDataset_train = get_dataset('face', config, train=True)

FaceDataset_test = get_dataset('face', config, train=False)
BodyDataset_test = get_dataset('body', config, train=False)
FrameDataset_test = get_dataset('frame', config, train=False)

# pdb.set_trace()

vis_test = BodyDataset_train.visualise(40000)
vis_test.save('face_test1.jpg')

print('done')