import json
from collections import defaultdict
from pathlib import Path

import cv2

from florence_2.read_interaction_csv import NOT_NAMES
from florence_2.show_results import get_all_annotation_paths, validate_data, COLORS_DICT


def show_florence2_results(video_file_paths, annotation_root):
    dict_results_path = Path(r'C:\Workspace\ChimpanzeesThesis\outputs\signal_frames_data')
    dict_results_path.mkdir(exist_ok=True, parents=True)
    results_dict = {}
    report = {}
    matching_frames_indexes = defaultdict(list)
    # for video_path in [Path('D:/per_signal_videos') / '26_7_2017_a__index_166.mp4']:
    for video_path in video_file_paths:
        results_dict[video_path.name] = {}
        case_info_path = video_path.parent / f'{video_path.stem}.json'
        case_info = json.loads(case_info_path.read_text())

        s_id = case_info['recipient_id']
        t_id = case_info['signaler_id']
        if s_id in NOT_NAMES or t_id in NOT_NAMES:
            print(f"Skipping {s_id}, {t_id}: {video_path.stem}")
            report[video_path.name] = 'unknown_individuals'
            continue

        # Open the video file
        video_capture = cv2.VideoCapture(video_path.as_posix())  # Replace with your video file path

        # Check if the video file was opened successfully
        if not video_capture.isOpened():
            print(f"Error opening video file: {video_path.stem}")
            report[video_path.name] = 'error_in_video_file'
            continue

        number_of_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_capture.release()

        is_video_annotation_present = True
        for annotation_type in annotation_root.iterdir():
            if annotation_type.name not in COLORS_DICT:
                continue
            annotation_path = annotation_type / video_path.stem
            if not annotation_path.exists() or len(list(annotation_path.iterdir())) != number_of_frames:
                is_video_annotation_present = False
                break

        if not is_video_annotation_present:
            print(f"missing_video_annotations: {video_path.stem}")
            report[video_path.name] = 'missing_video_annotations'
            continue

        for current_frame_ind in range(number_of_frames):
            all_annotation_file_paths = get_all_annotation_paths(annotation_root, current_frame_ind, video_path)

            current_frame_ind += 1

            if not all([f.exists() for f in all_annotation_file_paths.values()]):
                report[video_path.name] = 'missing_frame_annotations'
                break
            res_dicts = {n: json.loads(f.read_text()) for n, f in all_annotation_file_paths.items()}
            list_of_data = validate_data(res_dicts)

            if len(list_of_data) != 2 or len(list(filter(lambda data: len(data['face']), list_of_data))) != 2:
                continue

            results_dict[video_path.name][current_frame_ind] = list_of_data
            matching_frames_indexes[video_path.name].append(current_frame_ind)
            # (dict_results_path / 'results_dict_new.json').write_text(json.dumps(results_dict, indent=4))
            # break

        if video_path.name not in report:
            number_of_matching_frames = len(results_dict[video_path.name])
            if number_of_matching_frames:
                report[video_path.name] = f'found matches'
            else:
                report[video_path.name] = f'no matching frames'

    print(report)
    (dict_results_path / 'report.json').write_text(json.dumps(report, indent=4))
    (dict_results_path / 'matching_frames_indexes.json').write_text(json.dumps(matching_frames_indexes, indent=4))


def main():
    video_root_path = Path('D:/per_signal_videos')
    annotation_root = Path(r'C:\Workspace\ChimpanzeesThesis\outputs\florence2_by_signals__26_9_24\home\ubuntu\segment-anything-2\florence_2\output')

    video_file_paths = [path for path in video_root_path.iterdir() if path.suffix == '.mp4']
    show_florence2_results(video_file_paths, annotation_root)


def count_issues_in_signal_videos_annotations():
    dict_results_path = Path(r'C:\Workspace\ChimpanzeesThesis\outputs\signal_frames_data')
    values_list = list(json.loads((dict_results_path / 'report.json').read_text()).values())
    print({v: values_list.count(v) for v in set(values_list)})


if __name__ == '__main__':
    main()
    count_issues_in_signal_videos_annotations()
