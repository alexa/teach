#!/usr/bin/env python

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import glob
import json
import multiprocessing as mp
import os
from argparse import ArgumentParser
from datetime import datetime

from teach.eval.compute_metrics import aggregate_metrics
from teach.inference.inference_runner import InferenceRunner, InferenceRunnerConfig
from teach.logger import create_logger
from teach.utils import dynamically_load_class

logger = create_logger(__name__)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help='Base data directory containing subfolders "games" and "edh_instances',
    )
    arg_parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Images directory for episode replay output",
    )
    arg_parser.add_argument(
        "--use_img_file",
        dest="use_img_file",
        action="store_true",
        help="synchronous save images with model api use the image file instead of streaming image",
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store output files from playing EDH instances",
    )
    arg_parser.add_argument(
        "--split",
        type=str,
        default="valid_seen",
        choices=["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"],
        help="One of train, valid_seen, valid_unseen, test_seen, test_unseen",
    )
    arg_parser.add_argument(
        "--edh_instance_file",
        type=str,
        help="Run only on this EDH instance. Split must be set appropriately to find corresponding game file.",
    )
    arg_parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use")
    arg_parser.add_argument(
        "--max_init_tries",
        type=int,
        default=5,
        help="Max attempts to correctly initialize an instance before declaring it as a failure",
    )
    arg_parser.add_argument(
        "--max_traj_steps",
        type=int,
        default=1000,
        help="Max predicted trajectory steps",
    )
    arg_parser.add_argument("--max_api_fails", type=int, default=30, help="Max allowed API failures")
    arg_parser.add_argument(
        "--metrics_file",
        type=str,
        required=True,
        help="File used to store metrics",
    )
    arg_parser.add_argument(
        "--model_module",
        type=str,
        default="teach.inference.sample_model",
        help="Path of the python module to load the model class from.",
    )
    arg_parser.add_argument(
        "--model_class", type=str, default="SampleModel", help="Name of the TeachModel class to use during inference."
    )
    arg_parser.add_argument(
        "--replay_timeout", type=int, default=500, help="The timeout for playing back the interactions in an episode."
    )

    start_time = datetime.now()
    args, model_args = arg_parser.parse_known_args()

    if args.edh_instance_file:
        edh_instance_files = [args.edh_instance_file]
    else:
        inference_output_files = glob.glob(os.path.join(args.output_dir, "inference__*.json"))
        finished_edh_instance_files = [os.path.join(fn.split("__")[1]) for fn in inference_output_files]
        edh_instance_files = [
            os.path.join(args.data_dir, "edh_instances", args.split, f)
            for f in os.listdir(os.path.join(args.data_dir, "edh_instances", args.split))
            if f not in finished_edh_instance_files
        ]
        if not edh_instance_files:
            print(
                f"all the edh instances have been ran for input_dir={os.path.join(args.data_dir, 'edh_instances', args.split)}"
            )
            exit(1)

    runner_config = InferenceRunnerConfig(
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        metrics_file=args.metrics_file,
        num_processes=args.num_processes,
        max_init_tries=args.max_init_tries,
        max_traj_steps=args.max_traj_steps,
        max_api_fails=args.max_api_fails,
        model_class=dynamically_load_class(args.model_module, args.model_class),
        replay_timeout=args.replay_timeout,
        model_args=model_args,
        use_img_file=args.use_img_file,
    )

    runner = InferenceRunner(edh_instance_files, runner_config)
    metrics = runner.run()
    inference_end_time = datetime.now()
    logger.info("Time for inference: %s" % str(inference_end_time - start_time))

    results = aggregate_metrics(metrics, args)
    print("-------------")
    print(
        "SR: %d/%d = %.3f"
        % (
            results["success"]["num_successes"],
            results["success"]["num_evals"],
            results["success"]["success_rate"],
        )
    )
    print(
        "GC: %d/%d = %.3f"
        % (
            results["goal_condition_success"]["completed_goal_conditions"],
            results["goal_condition_success"]["total_goal_conditions"],
            results["goal_condition_success"]["goal_condition_success_rate"],
        )
    )
    print("PLW SR: %.3f" % (results["path_length_weighted_success_rate"]))
    print("PLW GC: %.3f" % (results["path_length_weighted_goal_condition_success_rate"]))
    print("-------------")

    results["traj_stats"] = metrics
    with open(args.metrics_file, "w") as h:
        json.dump(results, h)

    end_time = datetime.now()
    logger.info("Total time for inference and evaluation: %s" % str(end_time - start_time))


if __name__ == "__main__":
    # Using spawn method, parent process creates a new and independent child process,
    # which avoid sharing unnecessary resources.
    # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
    mp.set_start_method("spawn", force=True)
    main()
