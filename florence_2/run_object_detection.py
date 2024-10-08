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


def run_florence2_on_image_by_prompt(inferencer, video_path, data_name):
    output_folder = Path("./output") / data_name / video_path.stem
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
        res_dict = inferencer(Image.fromarray(frame))
        file_path.write_text(json.dumps(res_dict))

    # plot_bbox(image, results[task_prompt])


def main():
    model_id = 'microsoft/Florence-2-large'
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype='auto').eval().cuda()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    body_part_texts = ['face', 'head', 'ear']
    body_part_tasks = {element: element for element in body_part_texts}
    full_body_task = {'body': 'chimpanzee'}

    caption_to_phrase_groundings_by_name = {**body_part_tasks, **full_body_task}

    open_vocabulary_detection_texts = {
        'query_count': "how many chimpanzees are in the image?",
        'query_actions': "in case the chimpanzees interact with each other tell me what they do?"
    }

    task_prompts_with_inputs = {
        '<CAPTION_TO_PHRASE_GROUNDING>': caption_to_phrase_groundings_by_name,
        '<OPEN_VOCABULARY_DETECTION>': open_vocabulary_detection_texts,
        # '<DENSE_REGION_CAPTION>': {'dense_region_caption': None},
        # '<DETAILED_CAPTION>': {'detailed_caption': None},
        # '<MORE_DETAILED_CAPTION>': {'more_detailed_caption': None}
    }  # type: dict[str, dict[str, str]]

    # video_root_path = Path('/home/ubuntu/videos_2019/')
    video_root_path = Path('/home/ubuntu/per_signal_videos/')
    for video_path in video_root_path.iterdir():
        for task_prompt, text_inputs in task_prompts_with_inputs.items():
            for full_task_name, text_input in text_inputs.items():
                def run_florence2_on_image(image_array):
                    return run_florence2(model, processor, image_array, task_prompt, text_input)
                run_florence2_on_image_by_prompt(run_florence2_on_image, video_path, full_task_name)


if __name__ == '__main__':
    main()
