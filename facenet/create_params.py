import argparse

from pathlib import Path
# from clearml import Task


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=1000, help='number of epochs')
    return parser.parse_args()


if __name__ == '__main__':
    # task = Task.init(auto_connect_arg_parser=True)

    print(parse_opt())
    # task.connect(parse_opt().__dict__)

