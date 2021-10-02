#!/usr/bin/env python

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
import os
import tarfile

import boto3
from botocore import UNSIGNED
from botocore.config import Config

DEFAULT_DATASET_BUCKET_NAME = "teach-dataset"
DEFAULT_DIRECTORY = "/tmp/teach-dataset"
FILE_LIST = [
    "all_games.tar.gz",
    "edh_instances.tar.gz",
    "experiment_games.tar.gz",
    "images_and_states.tar.gz",
    "tfd_instances.tar.gz",
]


def download_dataset(directory, key=None, bucket_name=DEFAULT_DATASET_BUCKET_NAME):
    """
    Download file from the S3 bucket to the target directory.
    If key is not given, download all available files in the bucket.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        s3_resource = boto3.resource("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
        bucket = s3_resource.Bucket(bucket_name)
        if key:
            print(f"Downloading s3://{bucket_name}/{key} to {directory}")
            bucket.download_file(Key=key, Filename=f"{directory}/{key}")
        else:
            for file_name in FILE_LIST:
                print(f"Downloading s3://{bucket_name}/{file_name} to {directory}")
                bucket.download_file(Key=file_name, Filename=f"{directory}/{file_name}")
    except Exception as e:
        print(f"Exception reading from: {bucket_name}")
        print(f"Exception: {str(e)}")


def extract_dataset(directory, file_name=None):
    """
    Extract extract archive file(s) in the given directory.
    """
    print(f"Extracting dataset to {directory}")
    if file_name:
        with tarfile.open(os.path.join(directory, file_name)) as archive:
            archive.extractall(directory)
        print(f"Extracted file: {file_name}")
    else:
        for file_name in FILE_LIST:
            with tarfile.open(os.path.join(directory, file_name)) as archive:
                archive.extractall(directory)
            print(f"Extracted file: {file_name}")


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
    print("Input skip-download:", skip_extract)
    print("Input file:", file_name)

    if not skip_download:
        download_dataset(directory, key=file_name)
    if not skip_extract:
        extract_dataset(directory, file_name=file_name)


if __name__ == "__main__":
    main()
