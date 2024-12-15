import os
import sys
import time
from pathlib import Path


def main():
    print(f'start {time.strftime("%Y-%m-%d %H:%M:%S")}')
    # add values to arguments here, to be loaded inside using ArgumentParser

    sys.argv.append("--config_path")
    sys.argv.append(Path('2024-11-27.yaml').resolve().as_posix())

    os.environ["TRAIN_IMAGE_PATH"] = '.'
    os.environ["VAL_IMAGE_PATH"] = '.'

    os.environ["TRAIN_LABEL_PATH"] = '.'
    os.environ["VAL_LABEL_PATH"] = '.'

    import retinaface.train
    retinaface.train.main()
    print(f'end {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
