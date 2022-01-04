#!/usr/bin/env python

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
import os
import tarfile

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

DEFAULT_DATASET_BUCKET_NAME = "teach-dataset"
DEFAULT_DIRECTORY = "/tmp/teach-dataset"
FILE_LIST = [
    "all_games.tar.gz",
    "edh_instances.tar.gz",
    "experiment_games.tar.gz",
    "images_and_states.tar.gz",
    "tfd_instances.tar.gz",
    "baseline_models.tar.gz",
    "et_pretrained_models.tar.gz",
]


def update_download_progressbar(t):
    def inner(bytes_amount):
        t.update(bytes_amount)

    return inner


def download_with_progressbar(s3_resource, bucket_name, key, directory):
    file_object = s3_resource.Object(bucket_name=bucket_name, key=key)
    total_file_size = file_object.content_length
    bucket = s3_resource.Bucket(bucket_name)
    with tqdm(total=total_file_size, unit="B", unit_scale=True, desc=key) as t:
        bucket.download_file(Key=key, Filename=f"{directory}/{key}", Callback=update_download_progressbar(t))


def download_dataset(directory, key=None, bucket_name=DEFAULT_DATASET_BUCKET_NAME):
    """
    Download file from the S3 bucket to the target directory.
    If key is not given, download all available files in the bucket.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        s3_resource = boto3.resource("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
        if key:
            print(f"Downloading s3://{bucket_name}/{key} to {directory}")
            download_with_progressbar(s3_resource, bucket_name, key, directory)
        else:
            for file_name in FILE_LIST:
                print(f"Downloading s3://{bucket_name}/{file_name} to {directory}")
                download_with_progressbar(s3_resource, bucket_name, file_name, directory)
    except Exception as e:
        print(f"Exception reading from: {bucket_name}")
        print(f"Exception: {str(e)}")


def extract_all_with_progress(archive, directory):
    members = archive.getmembers()
    for member in tqdm(iterable=members, total=len(members)):
        archive.extract(member=member, path=directory)


def extract_dataset(directory, file_name=None):
    """
    Extract extract archive file(s) in the given directory.
    """
    print(f"Extracting dataset to {directory}")
    if file_name:
        print(f"Extracting file: {file_name}")
        with tarfile.open(os.path.join(directory, file_name)) as archive:
            extract_all_with_progress(archive, directory)

    else:
        for file_name in FILE_LIST:
            print(f"Extracting file: {file_name}")
            with tarfile.open(os.path.join(directory, file_name)) as archive:
                extract_all_with_progress(archive, directory)


def process_arguments():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-se",
        "--skip-extract",
        dest="skip_extract",
        action="store_true",
        required=False,
        help="If set, skip extracting archive files.",
    )
    group.add_argument(
        "-sd",
        "--skip-download",
        dest="skip_download",
        action="store_true",
        required=False,
        help="If set, skip downloading archive files.",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=False,
        default=DEFAULT_DIRECTORY,
        help=f"The location to store the dataset into. Default={DEFAULT_DIRECTORY}",
    )
    parser.add_argument(
        "-f", "--file", type=str, required=False, help="Specify the file name to be retrieved from S3 bucket."
    )

    args = parser.parse_args()
    return args


def main():
    args = process_arguments()

    directory = args.directory
    skip_extract = args.skip_extract
    skip_download = args.skip_download
    file_name = args.file

    print("Input directory:", directory)
    print("Input skip-extract:", skip_extract)
    print("Input skip-download:", skip_download)
    print("Input file:", file_name)

    if not skip_download:
        download_dataset(directory, key=file_name)
    if not skip_extract:
        extract_dataset(directory, file_name=file_name)


if __name__ == "__main__":
    main()
