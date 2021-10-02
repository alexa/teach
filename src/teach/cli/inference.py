#!/usr/bin/env python

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
import json
import os
from argparse import ArgumentParser

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
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store output files from playing EDH instances",
    )
    arg_parser.add_argument(
        "--split",
        type=str,
        default="valid_seen",
        choices=["train", "valid_seen", "valid_unseen"],
        help="One of train, valid_seen, valid_unseen",
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
        "model_args", nargs=argparse.REMAINDER, help="Any unknown arguments will be captured and passed to the model"
    )
    args = arg_parser.parse_args()

    if args.edh_instance_file:
        edh_instance_files = [args.edh_instance_file]
    else:
        edh_instance_files = [
            os.path.join(args.data_dir, "edh_instances", args.split, f)
            for f in os.listdir(os.path.join(args.data_dir, "edh_instances", args.split))
        ]

    runner_config = InferenceRunnerConfig(
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        metrics_file=args.metrics_file,
        num_processes=args.num_processes,
        max_init_tries=args.max_init_tries,
        max_traj_steps=args.max_traj_steps,
        max_api_fails=args.max_api_fails,
        model_class=dynamically_load_class(args.model_module, args.model_class),
        model_args=args.model_args,
    )

    runner = InferenceRunner(edh_instance_files, runner_config)
    metrics = runner.run()
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

    results["traj_metrics"] = metrics
    with open(args.metrics_file, "w") as h:
        json.dump(metrics, h)


if __name__ == "__main__":
    main()
