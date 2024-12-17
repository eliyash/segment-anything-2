import subprocess
import os
import json
from pathlib import Path

# import pdb

num_threads = 30

json_fid = open('config.json')
config = json.load(json_fid)

video_dir = os.path.join(config['dataset_dir'], 'videos')

# extract_command_fmt = \
#     'ffmpeg -i {video_path} -threads {num_threads} -vf scale={img_width}:-1 -vf fps={fps} -deinterlace -q:v 1 {frame_path}'

extract_command_fmt = \
    'C:/Portable/ffmpeg-4.4.1/bin/ffmpeg -i "{video_path}" -threads {num_threads} -deinterlace -q:v 1 "{frame_path}"'

videos_folder = Path(video_dir)
for root, dirs, files in os.walk(video_dir):
    for file in files:
        if file.endswith(".mp4"):
            video_path = Path(root) / file
            frame_path = Path(root.replace('videos', 'frames')) / file / config['img_fmt']
            extract_command = extract_command_fmt.format(video_path=video_path.as_posix(),
                                                             num_threads=num_threads,
                                                             #img_width=config['img_width'],
                                                             #fps=config['fps'],
                                                             frame_path=frame_path.as_posix())

            subprocess.call(extract_command, shell=True)