# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os.path import isdir
from pathlib import Path
from typing import List, Type

from PIL import Image

from teach.dataset.definitions import Definitions
from teach.dataset.interaction import Interaction
from teach.eval.compute_metrics import create_new_traj_metrics, evaluate_traj
from teach.inference.actions import obj_interaction_actions
from teach.inference.teach_model import TeachModel
from teach.logger import create_logger
from teach.replay.episode_replay import EpisodeReplay
from teach.utils import (
    create_task_thor_from_state_diff,
    load_images,
    save_dict_as_json,
    with_retry,
)

definitions = Definitions(version="2.0")
action_id_to_info = definitions.map_actions_id2info
logger = create_logger(__name__)


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


class InferenceRunner:
    def __init__(self, edh_instance_files, config: InferenceRunnerConfig):
        self._edh_instance_files = edh_instance_files
        self._config = config

    def run(self):
        self._launch_processes(self._edh_instance_files, self._config)
        return self._load_metrics()

    def _load_metrics(self):
        metrics = dict()
        for metrics_file in InferenceRunner._get_metrics_files(self._config):
            if os.path.isfile(metrics_file):
                with open(metrics_file) as h:
                    thread_replay_status = json.load(h)
                metrics.update(thread_replay_status)
        return metrics

    @staticmethod
    def _get_metrics_files(config):
        return [
            InferenceRunner._get_metrics_file_name_for_process(x, config.metrics_file)
            for x in range(config.num_processes)
        ]

    @staticmethod
    def _launch_processes(edh_instance_files, config: InferenceRunnerConfig):
        processes = []
        ers = []
        try:
            for process_index in range(config.num_processes):
                er = EpisodeReplay("thor", ["ego", "allo", "targetobject"])
                ers.append(er)
                process = InferenceRunner._launch_process(process_index, edh_instance_files, config, er)
                processes.append(process)
        finally:
            InferenceRunner._join_processes(processes)
            for er in ers:
                er.simulator.shutdown_simulator()

    @staticmethod
    def _launch_process(process_index, edh_instance_files, config: InferenceRunnerConfig, er: EpisodeReplay):
        num_files = len(edh_instance_files)
        num_files_per_process = InferenceRunner._get_num_files_per_process(
            num_files=num_files, num_processes=config.num_processes
        )
        start_index, end_index = InferenceRunner._get_range_to_process(
            process_index=process_index,
            num_files_per_process=num_files_per_process,
            num_files=num_files,
        )

        files_to_process = edh_instance_files[start_index:end_index]

        process = mp.Process(target=InferenceRunner._run, args=(process_index, files_to_process, config, er))

        process.start()
        time.sleep(0.1)
        return process

    @staticmethod
    def _run(process_index, files_to_process, config: InferenceRunnerConfig, er: EpisodeReplay):
        metrics_file = InferenceRunner._get_metrics_file_name_for_process(process_index, config.metrics_file)
        metrics = dict()

        model = config.model_class(process_index, config.num_processes, model_args=config.model_args)

        for file_index, instance_file in enumerate(files_to_process):
            try:
                instance_id, instance_metrics = InferenceRunner._run_edh_instance(instance_file, config, model, er)
                metrics[instance_id] = instance_metrics
                save_dict_as_json(metrics, metrics_file)

                logger.info(f"Instance {instance_id}, metrics: {instance_metrics}")
                logger.info(f"Process {process_index} completed {file_index + 1} / {len(files_to_process)} instances")
            except Exception:
                err_msg = f"exception happened for instance={instance_file}, continue with the rest"
                logger.error(err_msg, exc_info=True)
                continue

    @staticmethod
    def _load_edh_history_images(edh_instance, config: InferenceRunnerConfig):
        image_file_names = edh_instance["driver_image_history"]
        image_dir = os.path.join(config.data_dir, "images", config.split, edh_instance["game_id"])
        return load_images(image_dir, image_file_names)

    @staticmethod
    def _run_edh_instance(instance_file, config: InferenceRunnerConfig, model: TeachModel, er: EpisodeReplay):
        edh_instance = InferenceRunner._load_edh_instance(instance_file)

        edh_check_task = create_task_thor_from_state_diff(edh_instance["state_changes"])
        game_file = InferenceRunner._get_game_file(edh_instance, config)

        metrics = create_new_traj_metrics(edh_instance)
        instance_id = edh_instance["instance_id"]
        logger.debug(f"Processing instance {instance_id}")

        try:
            init_success, er = with_retry(
                fn=lambda: InferenceRunner._initialize_episode_replay(
                    edh_instance, game_file, edh_check_task, config.replay_timeout, er
                ),
                retries=config.max_init_tries - 1,
                check_first_return_value=True,
            )
        except Exception:
            init_success = False
            logger.error(f"Failed to initialize episode replay for instance={instance_id}", exc_info=True)

        edh_history_images = None
        try:
            if not config.use_img_file:
                edh_history_images = InferenceRunner._load_edh_history_images(edh_instance, config)
        except Exception:
            init_success = False
            logger.error(f"Failed to load_edh_history_images for {instance_id}", exc_info=True)

        metrics["init_success"] = init_success
        if not init_success:
            return edh_instance["instance_id"], metrics

        model_started_success = False
        try:
            model_started_success = model.start_new_edh_instance(edh_instance, edh_history_images, instance_file)
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
                    img = InferenceRunner._get_latest_ego_image(er)
                    image_name = InferenceRunner._save_image(config, edh_instance, img, traj_steps_taken)
                    action, obj_relative_coord = model.get_next_action(
                        img, edh_instance, prev_action, image_name, instance_file
                    )
                    step_success = InferenceRunner._execute_action(er.simulator, action, obj_relative_coord)
                    InferenceRunner._update_metrics(metrics, action, obj_relative_coord, step_success)
                    prev_action = {"action": action, "obj_relative_coord": obj_relative_coord}
                    pred_actions.append(prev_action)
                except Exception as e:
                    logger.error(
                        f"_run_edh_instance Exception: {str(e)} for instance_id={instance_id}, "
                        f"traj_steps_taken={traj_steps_taken}",
                        exc_info=True,
                    )
                    metrics["error"] = 1
                    break
                if InferenceRunner._should_end_inference(action, metrics, config.max_api_fails):
                    break

        (
            success,
            final_goal_conditions_total,
            final_goal_conditions_satisfied,
        ) = InferenceRunner._check_episode_progress(er, edh_check_task)

        metrics_diff = evaluate_traj(
            success,
            edh_instance,
            traj_steps_taken,
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
    def _initialize_episode_replay(edh_instance, game_file, task, replay_timeout, er: EpisodeReplay):
        start_time = time.perf_counter()
        er.set_episode_by_fn_and_idx(game_file, 0, 0)
        edh_interactions = list()
        for interaction in edh_instance["interactions"][: edh_instance["pred_start_idx"]]:
            action = action_id_to_info[interaction["action_id"]]
            edh_interactions.append(Interaction.from_dict(interaction, action["action_type"]))
        er.episode.interactions = edh_interactions

        init_success = False
        with ThreadPoolExecutor() as tp:
            future = tp.submit(er.play_episode, task=task, shutdown_on_finish=False)
            logger.info(f"Started episode replay with timeout: {replay_timeout} sec")
            init_success, _ = future.result(timeout=replay_timeout)

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Elapsed time for episode replay: {elapsed_time}")

        return init_success, er if init_success else None

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
    def _get_game_file(edh_instance, config: InferenceRunnerConfig):
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
    def _load_edh_instance(instance_file):
        with open(instance_file) as handle:
            edh_instance = json.load(handle)
        return edh_instance

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

    @staticmethod
    def _save_image(config, edh_instance, img, traj_steps_taken):
        image_name = f"img__{edh_instance['instance_id']}_{traj_steps_taken}.jpeg"
        if config.use_img_file:
            InferenceRunner._save_image_sync(img, image_name, config)
        else:
            InferenceRunner._save_image_async(img, image_name, config)
        return image_name

    @staticmethod
    def _save_image_async(img, image_name, config: InferenceRunnerConfig):
        process = mp.Process(target=InferenceRunner._save_image_sync, args=(img, image_name, config))
        process.start()
        return image_name

    @staticmethod
    def _save_image_sync(img, image_name, config: InferenceRunnerConfig):
        if not isdir(config.images_dir):
            Path(config.images_dir).mkdir(parents=True, exist_ok=True)
        image_path = os.path.join(config.images_dir, image_name)
        img.save(image_path)
        return image_name
