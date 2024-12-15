import subprocess
import os
import json
# import pdb

num_threads = 30

json_fid = open('config.json')
config = json.load(json_fid)

video_dir = os.path.join(config['dataset_dir'], 'videos')

# extract_command_fmt = \
#     'ffmpeg -i {video_path} -threads {num_threads} -vf scale={img_width}:-1 -vf fps={fps} -deinterlace -q:v 1 {frame_path}'

extract_command_fmt = \
    'ffmpeg -i {video_path} -threads {num_threads} -deinterlace -q:v 1 {frame_path}'

for root, dirs, files in os.walk(video_dir):
    for file in files:
        if file.endswith(".mp4"):
            video_path = os.path.join(root, file)
            frame_path = os.path.join(video_path.replace('videos', 'frames'), config['img_fmt'])
            extract_command = extract_command_fmt.format(video_path=video_path,
                                                             num_threads=num_threads,
                                                             #img_width=config['img_width'],
                                                             #fps=config['fps'],
                                                             frame_path=frame_path)

            subprocess.call(extract_command, shell=True)
