# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import copy
import importlib.resources
import json
import os
from collections import OrderedDict
from pathlib import Path

from teach import meta_data_files as meta_data_files
from teach.dataset.task_THOR import Task_THOR
from teach.meta_data_files import task_definitions


class Definitions:
    def __init__(self, definitions=None, simulator="THOR", version="2.0"):
        self.simulator = simulator
        self.version = version
        if definitions is None:
            with importlib.resources.open_text(meta_data_files, "default_definitions.json") as data_file:
                definitions = json.load(data_file, object_pairs_hook=OrderedDict)["definitions"]

            if version == "2.0" and simulator == "THOR":
                tasks, task_id_to_task_dict, task_name_to_task_dict = Task_THOR.load_tasks(task_definitions)
                definitions["tasks"] = tasks
                self.map_tasks_id2info = task_id_to_task_dict
                self.map_tasks_name2info = task_name_to_task_dict
            else:
                raise RuntimeError("No support for version " + str(version) + " with simulator " + str(simulator))

        self.info = definitions
        self.map_agents_id2info = self.__create_lookup_agents()
        self.map_status_id2name = self.__create_lookup_status()
        self.map_actions_id2info, self.map_actions_name2info = self.__create_lookup_actions()

    def __get_files_recursive(self, root_dir, file_list, extension=".json"):
        for path in Path(root_dir).iterdir():
            if path.is_dir():
                self.__get_files_recursive(path, file_list)
            elif os.path.isfile(path) and path.suffix == extension:
                file_list.append(path.resolve())

    def to_dict(self):
        info_dict = copy.deepcopy(self.info)
        info_dict["tasks"] = [x.to_dict() for x in info_dict["tasks"]]
        return info_dict

    def __create_lookup_actions(self):
        _map_id = OrderedDict()
        for action in self.info["actions"]:
            _map_id[action["action_id"]] = OrderedDict(
                [
                    ("action_name", action["action_name"]),
                    ("action_type", action["action_type"]),
                    ("pose", action.get("pose")),
                    ("pose_delta", action.get("pose_delta")),
                ]
            )
        _map_name = OrderedDict()
        for action in self.info["actions"]:
            _map_name[action["action_name"]] = OrderedDict(
                [
                    ("action_id", action["action_id"]),
                    ("action_type", action["action_type"]),
                    ("pose", action.get("pose")),
                    ("pose_delta", action.get("pose_delta")),
                ]
            )
        return _map_id, _map_name

    def __create_lookup_tasks(self):
        _map_id = OrderedDict()
        for task in self.info["tasks"]:
            _map_id[task["task_id"]] = OrderedDict(
                [
                    ("task_id", task["task_id"]),
                    ("task_name", task["task_name"]),
                    ("task_nparams", task["task_nparams"]),
                    ("subgoals", task["subgoals"]),
                ]
            )
        _map_name = OrderedDict()
        for task in self.info["tasks"]:
            _map_name[task["task_name"]] = OrderedDict(
                [
                    ("task_id", task["task_id"]),
                    ("task_name", task["task_name"]),
                    ("task_nparams", task["task_nparams"]),
                    ("subgoals", task["subgoals"]),
                ]
            )
        return _map_id, _map_name

    def __create_lookup_agents(self):
        _map = OrderedDict()
        for agent in self.info["agents"]:
            _map[agent["agent_id"]] = OrderedDict(
                [("agent_name", agent["agent_name"]), ("agent_type", agent["agent_type"])]
            )
        return _map

    def __create_lookup_status(self):
        _map = OrderedDict()
        for status in self.info["status"]:
            _map[status["status_id"]] = status["status_name"]
        return _map
