
IGNORE_NAME = 'negative'
UNKNOWN_NAME = 'UNKNOWN'

CHIMP_ID_NAMES = [
    'ben', 'glenn', 'gracie', 'jake', 'jean', 'jerrard', 'johari', 'julie', 'kima', 'nan', 'oliver', 'pandora',
    'regina', 'shaun', 'uki', 'yoshi', 'zoe', 'zuri'
]
CRR_NAMES = ['tua', 'peley', 'pama', 'velu', 'jeje', 'jire', 'fana', 'flanle', 'foaf', 'fanwa', 'fanle', 'joya', 'yo']
CHIMP_ID_NAME_TO_CLASS_INDEX = {name: i for i, name in enumerate(CHIMP_ID_NAMES)}
CRR_NAME_TO_CLASS_INDEX = {name: i+25 for i, name in enumerate(CRR_NAMES)}
ALL_NAMES_TO_CLASS_INDEX = {**CHIMP_ID_NAME_TO_CLASS_INDEX, **CRR_NAME_TO_CLASS_INDEX, 'OPEN_APE': 18, 'UNKNOWN': 19}
ALL_CLASS_INDEX_TO_NAMES = {v: k for k, v in ALL_NAMES_TO_CLASS_INDEX.items()}
