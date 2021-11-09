# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import logging
import os
from collections import OrderedDict

from teach.dataset.definitions import Definitions
from teach.dataset.task import Task
from teach.dataset.task_THOR import Task_THOR
from teach.logger import create_logger

logger = create_logger(__name__, logging.WARNING)


class Dataset:
    def __init__(self, task_type=None, definitions=None, comments="", version=None, tasks=None):
        self.version = "2.0" if version is None else version
        self.task_type = task_type
        self.comments = comments
        self.definitions = Definitions(definitions=definitions, version=version) if definitions is None else definitions
        self.tasks = tasks if tasks is not None else []

    def add_task(self, task):
        self.tasks.append(task)

    def to_dict(self):
        _dict = OrderedDict()
        _dict["version"] = self.version
        _dict["task_type"] = self.task_type
        _dict["comments"] = self.comments
        _dict["definitions"] = self.definitions.to_dict()
        _dict["tasks"] = [x.to_dict() for x in self.tasks]
        logger.info([type(x) for x in _dict["tasks"]])
        return _dict

    @classmethod
    def from_dict(cls, dataset_dict, process_init_state=True, version="2.0") -> "Dataset":
        definitions = Definitions(dataset_dict["definitions"])
        if version == "2.0":
            tasks = [
                Task_THOR.from_dict(task_dict, definitions, process_init_state)
                for task_dict in dataset_dict.get("tasks")
            ]
        else:
            tasks = [
                Task.from_dict(task_dict, definitions, process_init_state) for task_dict in dataset_dict.get("tasks")
            ]

        return cls(
            task_type=dataset_dict["task_type"],
            definitions=definitions,
            comments=dataset_dict.get("comments"),
            version=dataset_dict.get("version"),
            tasks=tasks,
        )

    def export_json(self, file_name, indent=4):
        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_name, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def import_json(cls, file_name, process_init_state=True, version="2.0") -> "Dataset":
        with open(file_name) as f:
            dataset_dict = json.load(f, object_pairs_hook=OrderedDict)
            return Dataset.from_dict(dataset_dict, process_init_state, version=version)
