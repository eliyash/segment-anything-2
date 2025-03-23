from pathlib import Path
from time import sleep
from typing import List

import labelbox as lb


API_KEY = (
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbHhodXVlb28x'
    'bGc0MDd4bWMxYTgwbHVjIiwib3JnYW5pemF0aW9uSWQiOiJja2k0enFnNmR0bzc4M'
    'Dc1NzFpNXV5OGdlIiwiYXBpS2V5SWQiOiJjbTY3dWZubGYwYTRnMDcwMmg4NDdodm'
    'F3Iiwic2VjcmV0IjoiMGU4ZTIzYTA3MDkyNjZlMTI3ODkyMDNjNjMyNTY1MjgiLCJ'
    'pYXQiOjE3Mzc1NDY2ODYsImV4cCI6MTc0MDEzODY4Nn0.vJ8M7S0beiB5l4YbVPUk'
    '-6X1BuzbIowSIiRrJb-ZtpI'
)


def upload_data(number_in_batch=10):
    images_folder = Path('D:/frames_collection_per_signal_fixed_interlacing_filtered')

    client = lb.Client(api_key=API_KEY)

    print('connecting', end='')
    while True:
        try:
            dataset = client.get_dataset("cm67uea0a001p0711tcqj9zbk")
            break
        except:
            print('.', end='')
            sleep(0.5)

    print('\nconnected')

    while True:
        existing_ids = {data_row.external_id for data_row in dataset.data_rows()}

        all_image_paths = [image_path for image_path in images_folder.iterdir() if image_path.name not in ["features.npy", "chosen.json"]]
        not_uploaded_image_paths = [image_path for image_path in all_image_paths if image_path.name not in existing_ids]

        if not len(not_uploaded_image_paths):
            break

        if len(not_uploaded_image_paths) > number_in_batch:
            not_uploaded_image_paths = not_uploaded_image_paths[:number_in_batch]

        image_data = []
        for image_path in not_uploaded_image_paths:
            print(f'uploading {image_path.name}')
            file_url = client.upload_file(image_path.as_posix())
            image_data.append({"row_data": file_url, "external_id": image_path.name})

        print(f'uploaded!')
        task = dataset.create_data_rows(image_data)
        task.wait_till_done()

        print(task.errors)
        print(f'iteration done')
    print('Done!!')


def upload_annoatations():
    client = lb.Client(api=API_KEY)
    dataset = client.get_dataset("cm67uea0a001p0711tcqj9zbk")
    print(dataset.uid)
    print(dataset.name)
    print(dataset.created_at)
    print(dataset.updated_at)
    print(dataset.description)
    print(dataset.project)
    print(dataset.data_rows())


def list_data(image_paths: List[Path]):
    image_data = []
    for file_url, image_path in image_paths:
        image_data.append({
            "row_data": file_url,
            "external_id": image_path.name
        })
    return image_data


if __name__ == '__main__':
    upload_data()

