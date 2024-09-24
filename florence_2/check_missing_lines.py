from collections import defaultdict
from pathlib import Path

import parse

from florence_2.read_interaction_csv import read_interation_data_as_list

MISMATCH_YEAR = {2007: 2017}
MISMATCH_FILE_NAME_TO_REAL_YEAR = {'7_2_18': 2019}


def get_date_data(file_name):
    match = parse.search('{}_{}_{}_', file_name.replace('(', '_').replace(' ', '_'))
    end_of_date = max([ve for vs, ve in match.spans.values()])
    n_day, n_month, n_year = map(int, match.fixed)
    n_year = n_year if n_year > 2000 else 2000 + n_year
    return n_day, n_month, n_year, end_of_date


def get_part_data(non_date_part, n_year):
    part_match = parse.search('({})', non_date_part)
    if not part_match:
        assert n_year == 2018
        part = non_date_part
    else:
        part = part_match.fixed[0]
    return part


def real_file_name_to_key_by_year(file_name: str, year: int):
    file_name = file_name.lower()
    n_day, n_month, n_year, end_of_date = get_date_data(file_name)
    non_date_part = file_name[end_of_date+1:]

    if int(n_year) != year:
        year_part = file_name[:end_of_date]
        if year_part in MISMATCH_FILE_NAME_TO_REAL_YEAR:
            n_year = MISMATCH_FILE_NAME_TO_REAL_YEAR[year_part]

    part = get_part_data(non_date_part, n_year)

    return n_day, n_month, n_year, part


def table_file_name_to_key_by_year(file_name: str, year: int):
    file_name = Path(file_name).stem.lower().removesuffix("_2")
    n_day, n_month, n_year, end_of_date = get_date_data(file_name)
    non_date_part = file_name[end_of_date+1:]
    n_year = MISMATCH_YEAR[n_year] if n_year in MISMATCH_YEAR else n_year

    part = get_part_data(non_date_part, n_year)

    return n_day, n_month, n_year, part


def match_videos_to_signals():
    video_root_path = Path('D:/')
    all_data = read_interation_data_as_list()

    all_video_keys = set()
    for year_folder in video_root_path.glob('videos_*'):
        year = int(year_folder.name.split('_')[1])
        for f in year_folder.glob('**/*'):
            if f.is_file() and f.suffix not in ['.docx', '.jpg']:
                all_video_keys.add(real_file_name_to_key_by_year(f.stem, year))

    videos_paths = {}
    for year_folder in video_root_path.glob('videos_*'):
        year = int(year_folder.name.split('_')[1])
        for f in year_folder.glob('**/*'):
            if f.is_file() and f.suffix not in ['.docx', '.jpg']:
                videos_paths[real_file_name_to_key_by_year(f.stem, year)] = f

    per_video_parts = defaultdict(list)
    for data in all_data:
        table_key = table_file_name_to_key_by_year(data['file_name'], data['year'])
        video_path = videos_paths[table_key]
        data['key'] = table_key
        data['real_path'] = video_path.relative_to(video_root_path).as_posix()
        per_video_parts[video_path].append(data)

    return per_video_parts


if __name__ == '__main__':
    match_videos_to_signals()
