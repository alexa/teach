# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import logging
import sys
from argparse import ArgumentParser
from io import BytesIO
from typing import List

import requests

from teach.inference.teach_model import TeachModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

TEACH_MODEL_API_URL_PREDICT = "http://{}/get_next_action"
TEACH_MODEL_API_URL_START_EDH = "http://{}/start_new_edh_instance"
TEACH_MODEL_API_URL_TEST = "http://{}/test"


class RemoteModelException(Exception):
    def __init__(self, message):
        super().__init__(message)


def assign_api_by_process_idx(host_and_ports, process_index):
    splits = host_and_ports.split(",")
    if process_index >= len(splits):
        raise RemoteModelException(f"process_index={process_index} can't be handled by available APIs:{splits}")
    return splits[process_index].strip()


class RemoteModel(TeachModel):
    def __init__(self, process_index: int, num_processes: int, model_args: List[str]):

        parser = ArgumentParser()
        parser.add_argument(
            "--model_api_host_and_port",
            type=str,
            default="localhost:5000",
            help="Teach Model API hosts and ports, E.g.:api1:5000,api2:5000",
        )
        args = parser.parse_args(model_args)

        host_and_port = assign_api_by_process_idx(args.model_api_host_and_port, process_index)
        self.test_url = TEACH_MODEL_API_URL_TEST.format(host_and_port)
        self.predict_url = TEACH_MODEL_API_URL_PREDICT.format(host_and_port)
        self.start_edh_url = TEACH_MODEL_API_URL_START_EDH.format(host_and_port)

    def get_next_action(self, img, edh_instance, prev_action, img_name=None, edh_name=None):
        if not img or not edh_instance:
            logger.warning("either img or edh_instance is None")
            return None, None
        img_in_memory = BytesIO()
        img.save(img_in_memory, "jpeg")
        img_in_memory.seek(0)
        data = {
            "img_name": img_name,
            "edh_name": edh_name,
            "prev_action": json.dumps(prev_action) if prev_action else None,
            "edh_instance": json.dumps(edh_instance),
        }

        resp = requests.post(self.predict_url, data=data, files={"img": (img_name, img_in_memory, "image/jpeg")})

        if resp.status_code != 200:
            logger.debug(f"failed sending data={data}")
            raise RemoteModelException(resp.text)

        resp_json = resp.json()
        action = resp_json.get("action")
        obj_relative_coord = resp_json.get("obj_relative_coord")
        return action, obj_relative_coord

    def test_connection(self):
        resp = requests.get(self.test_url)
        return resp.status_code == 200

    def start_new_edh_instance(self, edh_instance, edh_history_images, edh_name=None):
        images = []
        if edh_history_images:
            idx = 0
            for image in edh_history_images:
                img_in_memory = BytesIO()
                image.save(img_in_memory, "jpeg")
                img_in_memory.seek(0)
                images.append(("edh_history_images", (f"history{idx}", img_in_memory, "image/jpeg")))
                idx += 1

        data = {"edh_name": edh_name, "edh_instance": json.dumps(edh_instance)}
        resp = requests.post(self.start_edh_url, data=data, files=images)

        if resp.status_code != 200:
            logger.debug(f"failed sending data={data}")
            raise RemoteModelException(resp.text)

        return True
