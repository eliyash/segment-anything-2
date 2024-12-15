#!/usr/bin/env bash
#Parse Variables
DATA_DIR_STR=`grep -Po '"dataset_dir":(\d*?,|.*?[^\\]]",)' config.json`
DATA_DIR=$(echo $DATA_DIR_STR| cut -d'"' -f 4)

IM_WIDTH_STR=`grep -Po '"im_width":(\d*?,|.*?[^\\]"],)' config.json`
IM_WIDTH=$(echo $IM_WIDTH_STR| cut -d'"' -f 4)

FPS_STR=`grep -Po '"fps":(\d*?,|.*?[^\\]"],)' config.json`
FPS=$(echo $FPS_STR| cut -d'"' -f 3)

echo $DATA_DIR
echo $FPS
echo $IM_WIDTH
# Get list of video paths
VIDEO_DIR="$DATA_DIR/videos"
VIDEO_PTH_LIST=`find $VIDEO_DIR -name '*.mp4'`


# Extract frames
#for p in $VIDEO_PTH_LIST
#do
#    FRAME_DIR="${p/videos/frames}"
#    mkdir -p $FRAME_DIR
#    FRAME_PATH="$FRAME_DIR/%08d.jpg"
#    ffmpeg -i $p -threads 1 -vf scale=$IM_WIDTH:-1  -vf fps=$FPS -deinterlace -q:v 1 $FRAME_PATH
#
#