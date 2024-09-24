from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

MORE_THEN_ONE = 'MULTIPLE'
NO_ONE = 'NONE (IN VIEW)'
NOT_NAMES = [MORE_THEN_ONE, NO_ONE]


def time_str_to_second(time_str):
    # convert mm:ss.ms to seconds
    t = datetime.strptime(time_str, '%M:%S.%f')
    # Convert to timedelta and get total seconds
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond).total_seconds()


def read_interation_data_as_list():
    interation_data = Path(r"C:\Workspace\ChimpanzeesThesis\Chimpanzee ID Data\Video ID Instances Info.csv").read_text().splitlines()
    names = interation_data[0]
    # line format : 'Begin Time - hh:mm:ss.ms,End Time - hh:mm:ss.ms,Duration - hh:mm:ss.ms,RecipientID,SignalerID,File,YEAR,Signal Modality (FE or GE),'

    lines_data = []
    for index_in_sheet, interation_line in enumerate(interation_data[1:]):
        start_time, end_time, duration, recipient_id, signaler_id, file_name, year, modality, *_ = interation_line.split(',')

        start_seconds, end_seconds, duration_seconds = [time_str_to_second(t) for t in [start_time, end_time, duration]]
        params = {
            'recipient_id': recipient_id,
            'signaler_id': signaler_id,
            'start_seconds': start_seconds,
            'end_seconds': end_seconds,
            'index_in_sheet': index_in_sheet,
            'file_name': file_name,
            'year': int(year),
            'modality': modality
        }
        lines_data.append(params)
    return lines_data


def read_interation_data():
    interactions_per_video = defaultdict(list)
    for params in read_interation_data_as_list():
        interactions_per_video[params['file_name'].split('.')[0].lower()].append(params)
    return interactions_per_video


if __name__ == '__main__':
    print(read_interation_data())
