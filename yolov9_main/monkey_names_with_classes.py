
IGNORE_NAME = 'negative'
UNKNOWN_NAME = 'UNKNOWN'

CHIMP_ID_NAMES = [
    'ben', 'glenn', 'gracie', 'jake', 'jean', 'jerrard', 'johari', 'julie', 'kima', 'nan', 'oliver', 'pandora',
    'regina', 'shaun', 'uki', 'yoshi', 'zoe', 'zuri'
]
CRR_NAMES = ['tua', 'peley', 'pama', 'velu', 'jeje', 'jire', 'fana', 'flanle', 'foaf', 'fanwa', 'fanle', 'joya', 'yo']

CHIMP_ID_NAME_TO_CLASS_INDEX = {name: i for i, name in enumerate(CHIMP_ID_NAMES)}
OTHERS = {name: i+18 for i, name in enumerate(['OPEN_APE', 'UNKNOWN'])}
CRR_NAME_TO_CLASS_INDEX = {name: i+25 for i, name in enumerate(CRR_NAMES)}
VIDEO_FRAMES_DATASET = {name: i+50 for i, name in enumerate(['CHIMP_BODY', 'CHIMP_FACE', 'CHIMP_HEAD_FRONTAL', 'CHIMP_HEAD_BACK', 'CHIMP_HEAD_PROFILE', 'CHIMP_HEAD_ANGLED', 'CHIMP_HEAD_OCCLUDED_OR_BLURRY'])}
ALL_NAMES_TO_CLASS_INDEX = {**CHIMP_ID_NAME_TO_CLASS_INDEX, **CRR_NAME_TO_CLASS_INDEX, **OTHERS, **VIDEO_FRAMES_DATASET}
ALL_CLASS_INDEX_TO_NAMES = {v: k for k, v in ALL_NAMES_TO_CLASS_INDEX.items()}
