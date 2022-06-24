# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import time
from concurrent.futures import ThreadPoolExecutor

from teach.dataset.definitions import Definitions
from teach.dataset.interaction import Interaction
from teach.inference.inference_runner_base import InferenceRunnerConfig, InferenceRunnerBase
from teach.logger import create_logger
from teach.replay.episode_replay import EpisodeReplay
from teach.utils import create_task_thor_from_state_diff, load_images

definitions = Definitions(version="2.0")
action_id_to_info = definitions.map_actions_id2info
logger = create_logger(__name__)


class EdhInferenceRunner(InferenceRunnerBase):
    def __init__(self, input_files, config: InferenceRunnerConfig):
        super().__init__(input_files, config)

    @staticmethod
    def _load_edh_history_images(edh_instance, config: InferenceRunnerConfig):
        image_file_names = edh_instance['driver_image_history']
        image_dir = os.path.join(config.data_dir, 'images', config.split, edh_instance['game_id'])
        return load_images(image_dir, image_file_names)

    @staticmethod
    def _get_check_task(instance: dict, config: InferenceRunnerConfig):
        edh_check_task = create_task_thor_from_state_diff(instance["state_changes"])
        return edh_check_task

    @staticmethod
    def _get_instance_id(instance_file, instance):
        return instance['instance_id']

    @staticmethod
    def _maybe_load_history_images(instance: dict, config: InferenceRunnerConfig):
        """
        Load EDH history images
        :param instance: EDH instance
        :param config: InferenceRunnerConfig required for data directory
        """
        edh_history_images = None
        success = True
        try:
            if not config.use_img_file:
                edh_history_images = EdhInferenceRunner._load_edh_history_images(instance, config)
        except Exception:
            instance_id = instance['instance_id']
            logger.exception(f"Failed to load_edh_history_images for {instance_id}")
            success = False
        return success, edh_history_images

    @staticmethod
    def _initialize_episode_replay(instance, game_file, task, replay_timeout, er: EpisodeReplay):
        start_time = time.perf_counter()
        er.set_episode_by_fn_and_idx(game_file, 0, 0)
        edh_interactions = list()
        for interaction in instance["interactions"][:instance["pred_start_idx"]
        ]:
            action = action_id_to_info[interaction["action_id"]]
            edh_interactions.append(
                Interaction.from_dict(interaction, action["action_type"])
            )
        er.episode.interactions = edh_interactions

        with ThreadPoolExecutor() as tp:
            future = tp.submit(er.play_episode, task=task, shutdown_on_finish=False)
            logger.info(f"Started episode replay with timeout: {replay_timeout} sec")
            init_success, _ = future.result(timeout=replay_timeout)

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Elapsed time for episode replay: {elapsed_time}")

        return init_success, er if init_success else None
