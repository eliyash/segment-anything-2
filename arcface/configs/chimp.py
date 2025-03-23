from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r18"
config.resume = False
config.output = "D:/training_output/arcface"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 0.05
config.batch_size = 32
config.lr = 0.005
config.verbose = 2000
config.dali = False

config.rec = "D:/faces_dataset"
config.fine_tune = "C:/Users/Eliahu/Downloads/ms1mv3_arcface_r18_fp16"
# config.rec = "D:/arcface_files/face_dataset_train_only"
config.num_classes = 30
config.num_image = 2337
config.num_epoch = 3000
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]

config.num_workers = 0
