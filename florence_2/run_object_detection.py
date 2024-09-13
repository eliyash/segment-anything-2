import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import cv2


def run_florence2(model, processor, image, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to('cuda', torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )

    return parsed_answer


def run_florence2_on_image_by_prompt(video_root_path, task_prompt, text_input):
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def run_florence2_on_image(image_array):
        return run_florence2(model, processor, image_array, task_prompt, text_input)

    for video_path in video_root_path.iterdir():

        output_folder = Path("output") / 'florence2' / text_input / video_path.stem
        output_folder.mkdir(exist_ok=True, parents=True)

        # Open the video file
        video_capture = cv2.VideoCapture(video_path.as_posix())  # Replace with your video file path

        # Check if the video file was opened successfully
        if not video_capture.isOpened():
            print("Error opening video file")

        # Loop through the video frames
        i = 0
        while True:
            # Read the next frame
            ret, frame = video_capture.read()
            # If there are no more frames, break the loop
            if not ret:
                break
            file_path = output_folder / f'bboxs_{i}.json'
            i += 1
            if file_path.exists():
                continue
            res_dict = run_florence2_on_image(Image.fromarray(frame))
            file_path.write_text(json.dumps(res_dict))

    # plot_bbox(image, results[task_prompt])


def main():
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'

    video_root_path = Path('/home/ubuntu/videos_2019/')
    for body_part in ['ear', 'eye', 'nose', 'mouth', 'face']:
        run_florence2_on_image_by_prompt(video_root_path, task_prompt, body_part)
    run_florence2_on_image_by_prompt(video_root_path, task_prompt, "chimpanzee")


if __name__ == '__main__':
    main()
