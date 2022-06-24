# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import multiprocessing as mp
import os
import re
import time
from dataclasses import dataclass
from os.path import isdir
from pathlib import Path
from typing import List, Type
from abc import abstractmethod
from enum import Enum

from PIL import Image

from teach.eval.compute_metrics import create_new_traj_metrics, evaluate_traj
from teach.inference.actions import obj_interaction_actions
from teach.inference.teach_model import TeachModel
from teach.logger import create_logger
from teach.replay.episode_replay import EpisodeReplay
from teach.utils import save_dict_as_json, with_retry, load_json

logger = create_logger(__name__)


class InferenceBenchmarks(str, Enum):
    """Specify which TEACh benchmark to run inference on"""
    EDH = "edh"
    TFD = "tfd"


@dataclass
class InferenceRunnerConfig:
    data_dir: str
    split: str
    output_dir: str
    images_dir: str
    model_class: Type[TeachModel]
    model_args: List[str]
    metrics_file: str = "metrics.json"
    num_processes: int = 1
    max_init_tries: int = 3
    max_traj_steps: int = 1000
    max_api_fails: int = 30
    use_img_file: bool = False
    replay_timeout: int = 500


class InferenceRunnerBase:
    def __init__(self, input_files, config: InferenceRunnerConfig):
        self._input_files = input_files
        self._config = config

    def run(self):
        self._launch_processes(self._input_files, self._config)
        return self._load_metrics()

    def _load_metrics(self):
        metrics = dict()
        for metrics_file in InferenceRunnerBase._get_metrics_files(self._config):
            if os.path.isfile(metrics_file):
                with open(metrics_file) as h:
                    thread_replay_status = json.load(h)
                metrics.update(thread_replay_status)
        return metrics

    @staticmethod
    def _get_metrics_files(config):
        return [
            InferenceRunnerBase._get_metrics_file_name_for_process(x, config.metrics_file)
            for x in range(config.num_processes)
        ]

    def _launch_processes(self, input_files, config: InferenceRunnerConfig):
        processes = []
        ers = []
        try:
            for process_index in range(config.num_processes):
                er = EpisodeReplay("thor", ["ego", "allo", "targetobject"])
                ers.append(er)
                process = self._launch_process(process_index, input_files, config, er)
                processes.append(process)
        finally:
            InferenceRunnerBase._join_processes(processes)
            for er in ers:
                er.simulator.shutdown_simulator()

    def _launch_process(self, process_index, input_files, config: InferenceRunnerConfig, er: EpisodeReplay):
        num_files = len(input_files)
        num_files_per_process = InferenceRunnerBase._get_num_files_per_process(
            num_files=num_files, num_processes=config.num_processes
        )
        start_index, end_index = InferenceRunnerBase._get_range_to_process(
            process_index=process_index,
            num_files_per_process=num_files_per_process,
            num_files=num_files,
        )

        files_to_process = input_files[start_index:end_index]
        process = mp.Process(target=self._run, args=(process_index, files_to_process, config, er))

        process.start()
        time.sleep(0.1)
        return process

    @abstractmethod
    def _get_instance_id(self, instance_file, instance):
        """
        Return instance ID
        """
        raise NotImplementedError("Subclass must support returning instance ID")

    def _run(self, process_index, files_to_process, config: InferenceRunnerConfig, er: EpisodeReplay):
        metrics_file = InferenceRunnerBase._get_metrics_file_name_for_process(process_index, config.metrics_file)
        metrics = dict()

        model = config.model_class(process_index, config.num_processes, model_args=config.model_args)

        for file_index, instance_file in enumerate(files_to_process):
            try:
                instance_id, instance_metrics = self._run_instance(instance_file, config, model, er)
                metrics[instance_id] = instance_metrics
                save_dict_as_json(metrics, metrics_file)

                logger.info(f"Instance {instance_id}, metrics: {instance_metrics}")
                logger.info(f"Process {process_index} completed {file_index + 1} / {len(files_to_process)} instances")
            except Exception:
                err_msg = f"exception happened for instance={instance_file}, continue with the rest"
                logger.error(err_msg, exc_info=True)
                continue

    @abstractmethod
    def _get_check_task(self, instance: dict, config: InferenceRunnerConfig):
        """
        Creates Task_THOR object containing evaluation conditions
        :param instance: EDH or TfD instance
        :param config: InferenceRunnerConfig required for data directory
        """
        raise NotImplementedError("Subclass must implement method to create Task_THOR object containing evaluation "
                                  "conditions")

    @abstractmethod
    def _maybe_load_history_images(self, instance: dict, config: InferenceRunnerConfig):
        """
        Load history images if any
        :param instance: EDH or TfD instance
        :param config: InferenceRunnerConfig required for data directory
        """
        raise NotImplementedError("Subclass must implement method to load history images")

    def _run_instance(self, instance_file, config: InferenceRunnerConfig, model: TeachModel, er: EpisodeReplay):
        instance = load_json(instance_file)
        check_task = self._get_check_task(instance, config)

        game_file = InferenceRunnerBase.get_game_file(instance, config)
        game_id = re.sub(".game.json", "", os.path.basename(game_file))
        instance_id = self._get_instance_id(instance_file, instance)
        metrics = create_new_traj_metrics(instance_id, game_id)

        logger.debug(f"Processing instance {instance_id}")

        try:
            init_success, er = with_retry(
                fn=lambda: self._initialize_episode_replay(instance, game_file, check_task,
                                                           config.replay_timeout, er),
                retries=config.max_init_tries - 1,
                check_first_return_value=True,
            )
        except Exception:
            init_success = False
            logger.error(f"Failed to initialize episode replay for instance={instance_id}", exc_info=True)

        if "expected_init_goal_conditions_total" in instance and "expected_init_goal_conditions_satisfied" in instance:
            init_gc_total = instance["expected_init_goal_conditions_total"]
            init_gc_satisfied = instance["expected_init_goal_conditions_satisfied"]
        else:
            # For TfD instances, goal conditions are not cached so need an initial check
            (
                _,
                init_gc_total,
                init_gc_satisfied,
            ) = InferenceRunnerBase._check_episode_progress(er, check_task)

        history_load_success, history_images = self._maybe_load_history_images(instance, config)
        init_success = init_success and history_load_success

        metrics["init_success"] = init_success
        if not init_success:
            return instance_id, metrics

        model_started_success = False
        try:
            model_started_success = model.start_new_edh_instance(instance, history_images, instance_file)
        except Exception:
            model_started_success = False
            metrics["error"] = 1
            logger.error(f"Failed to start_new_edh_instance for {instance_id}", exc_info=True)

        if model_started_success:
            prev_action = None
            er.simulator.is_record_mode = True
            pred_actions = list()

            traj_steps_taken = 0
            for _ in range(config.max_traj_steps):
                traj_steps_taken += 1
                try:
                    img = InferenceRunnerBase._get_latest_ego_image(er)
                    image_name = self._save_image(config, instance_file, instance, img, traj_steps_taken)
                    action, obj_relative_coord = model.get_next_action(
                        img, instance, prev_action, image_name, instance_file)
                    step_success = InferenceRunnerBase._execute_action(er.simulator, action, obj_relative_coord)
                    InferenceRunnerBase._update_metrics(metrics, action, obj_relative_coord, step_success)
                    prev_action = {"action": action, "obj_relative_coord": obj_relative_coord}
                    pred_actions.append(prev_action)
                except Exception as e:
                    logger.error(
                        f"_run_instance Exception: {str(e)} for instance_id={instance_id}, "
                        f"traj_steps_taken={traj_steps_taken}",
                        exc_info=True,
                    )
                    metrics["error"] = 1
                    break
                if InferenceRunnerBase._should_end_inference(action, metrics, config.max_api_fails):
                    break

        (
            success,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        ) = InferenceRunnerBase._check_episode_progress(er, check_task)

        metrics_diff = evaluate_traj(
            success,
            instance,
            traj_steps_taken,
            init_gc_total,
            init_gc_satisfied,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        )
        metrics.update(metrics_diff)

        os.makedirs(config.output_dir, exist_ok=True)
        pred_actions_file = os.path.join(config.output_dir, "pred_actions__" + instance_id + ".json")
        with open(pred_actions_file, "w") as handle:
            json.dump(pred_actions, handle)

        er.simulator.dir_out = config.output_dir
        output_file = os.path.join(config.output_dir, "inference__" + instance_id + ".json")
        er.simulator.save(file_name=output_file)

        return instance_id, metrics

    @staticmethod
    def _check_episode_progress(er, task):
        (
            _,
            success,
            _,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        ) = er.simulator.check_episode_progress(task)
        return success, final_goal_conditions_total, final_goal_conditions_satisfied

    @staticmethod
    @abstractmethod
    def _initialize_episode_replay(instance, game_file, task, replay_timeout, er: EpisodeReplay):
        """
        Load initial state and replay history interactions if any
        """
        raise NotImplementedError("Subclass must specify initialization of inference per instance")

    @staticmethod
    def _get_latest_ego_image(er):
        return Image.fromarray(er.simulator.get_latest_images()["ego"])

    @staticmethod
    def _execute_action(simulator, action, obj_relative_coord):
        if action == "Stop":
            return True

        if action in obj_interaction_actions:
            y = obj_relative_coord[0]
            x = obj_relative_coord[1]
            step_success, _, _ = simulator.apply_object_interaction(action, 1, x, y)
            return step_success

        step_success, _, _ = simulator.apply_motion(action, 1)
        return step_success

    @staticmethod
    def get_game_file(edh_instance, config: InferenceRunnerConfig):
        return os.path.join(
            config.data_dir,
            "games",
            config.split,
            f"{edh_instance['game_id']}.game.json",
        )

    @staticmethod
    def _get_metrics_file_name_for_process(process_index, metrics_file):
        return f"{metrics_file}.json.{process_index}"

    @staticmethod
    def _update_metrics(metrics, action, obj_relative_coord, step_success):
        metrics["pred_actions"].append((action, obj_relative_coord))

        if action == "Stop":
            metrics["predicted_stop"] = 1

        if not step_success:
            metrics["num_api_fails"] += 1

    @staticmethod
    def _should_end_inference(action, metrics, max_api_fails):
        return action == "Stop" or metrics["num_api_fails"] >= max_api_fails

    @staticmethod
    def _get_range_to_process(process_index, num_files_per_process, num_files):
        start_index = process_index * num_files_per_process
        end_index = min(start_index + num_files_per_process, num_files)
        return start_index, end_index

    @staticmethod
    def _get_num_files_per_process(num_files, num_processes):
        return int(num_files / num_processes) + 1

    @staticmethod
    def _join_processes(processes):
        for process in processes:
            process.join()

    def _save_image(self, config, instance_file, instance, img, traj_steps_taken):
        image_name = f"img__{self._get_instance_id(instance_file, instance)}_{traj_steps_taken}.jpeg"
        if config.use_img_file:
            InferenceRunnerBase._save_image_sync(img, image_name, config)
        else:
            InferenceRunnerBase._save_image_async(img, image_name, config)
        return image_name

    @staticmethod
    def _save_image_async(img, image_name, config: InferenceRunnerConfig):
        process = mp.Process(target=InferenceRunnerBase._save_image_sync, args=(img, image_name, config))
        process.start()
        return image_name

    @staticmethod
    def _save_image_sync(img, image_name, config: InferenceRunnerConfig):
        if not isdir(config.images_dir):
            Path(config.images_dir).mkdir(parents=True, exist_ok=True)
        image_path = os.path.join(config.images_dir, image_name)
        img.save(image_path)
        return image_name
