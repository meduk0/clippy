import os
from argparse import ArgumentParser # Handles command-line arguments
from functools import partial
from multiprocessing.pool import ThreadPool # for the mutlitherading
from pathlib import Path
from urllib.request import urlretrieve # retrieve and save files
import pandas as pd #Used to read the .tsv (needed to extact image URL)

# the needed variables
DOWNLOAD_URL = "https://unsplash-datasets.s3.amazonaws.com/lite/latest/unsplash-research-dataset-lite-latest.zip"
# Fix: Define paths relative to the current file
DATASET_PATH = Path("unsplash-dataset")
DOWNLOADED_PHOTOS_PATH = Path("static/img")

def download_photo(image_width, photo):
    photo_id = photo[0]
    photo_url = photo[1] + f"?w={image_width}"
    photo_path = DOWNLOADED_PHOTOS_PATH / f"{photo_id}.jpg"
    print(f"Attempting to download {photo_id} to {photo_path}")
    if not photo_path.exists():
        try:
            urlretrieve(photo_url, photo_path)
            print(f"Successfully downloaded {photo_id}.jpg")
        except Exception as e:
            print(f"Cannot download {photo_url}: {e}")
    else:
        print(f"File {photo_id}.jpg already exists, skipping")

#args with default
def setup_parser():
    parser = ArgumentParser()
    parser.add_argument("--image_width", type=int, default=480)
    parser.add_argument("--threads_count", type=int, default=32)
    return parser

def main():
    # Parse arguments
    parser = setup_parser()
    args = parser.parse_args()
    os.system(f"curl -L {DOWNLOAD_URL} ./{DATASET_PATH}" )
    zip_filename = "unsplash-dataset.zip"
    print(f"Extracting {zip_filename}...")

    # Ensure the directory exists before extracting & extarcting the zip
    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    os.system(f"unzip {zip_filename} -d {str(DATASET_PATH)}")

    # reading the CSV to get each image URL link
    df = pd.read_csv(DATASET_PATH / "photos.tsv000", sep="\t", usecols=["photo_id", "photo_image_url"])
    photos = df.values.tolist()

    # Ensure the download directory exists
    DOWNLOADED_PHOTOS_PATH.mkdir(parents=True, exist_ok=True)
    #downloading the images using multitherads to speed up the process
    print("Photo downloading begins...")
    pool = ThreadPool(args.threads_count)
    pool.map(partial(download_photo, args.image_width), photos)
    print("Photo downloading finished!")

if __name__ == "__main__":
    main()
