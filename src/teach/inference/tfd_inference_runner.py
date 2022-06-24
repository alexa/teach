# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import re
import copy

from teach.dataset.definitions import Definitions
from teach.inference.inference_runner_base import InferenceRunnerConfig, InferenceRunnerBase
from teach.logger import create_logger
from teach.replay.episode_replay import EpisodeReplay
from teach.utils import save_dict_as_json, with_retry, load_images, load_json

definitions = Definitions(version="2.0")
task_id_to_task = definitions.map_tasks_id2info
logger = create_logger(__name__)


class TfdInferenceRunner(InferenceRunnerBase):
    def __init__(self, edh_instance_files, config: InferenceRunnerConfig):
        super().__init__(edh_instance_files, config)

    @staticmethod
    def _get_check_task(instance: dict, config: InferenceRunnerConfig):
        game_file = InferenceRunnerBase.get_game_file(instance, config)
        game = load_json(game_file)
        game_task_id = game["tasks"][0]["task_id"]
        task_to_check = copy.deepcopy(task_id_to_task[game_task_id])
        task_params = game["tasks"][0]["task_params"]
        task_to_check.task_params = task_params
        return task_to_check

    @staticmethod
    def _maybe_load_history_images(instance: dict, config: InferenceRunnerConfig):
        """
        TfD has no history so return an empty list
        :param instance: TfD instance
        :param config: InferenceRunnerConfig required for data directory
        """
        return True, list()

    @staticmethod
    def _get_instance_id(instance_file, instance):
        return re.sub(".json", "", os.path.basename(instance_file))

    @staticmethod
    def _initialize_episode_replay(instance, game_file, task, replay_timeout, er: EpisodeReplay):
        er.set_episode_by_fn_and_idx(game_file, 0, 0)
        er.episode.interactions = list()
        api_success, _ = er.set_up_new_episode(task=task)
        return api_success, er
