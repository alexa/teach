#!/usr/bin/env python

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import os
import re
from argparse import ArgumentParser

from teach.eval.compute_metrics import (
    aggregate_metrics,
    create_new_traj_metrics,
    load_traj_metrics,
)
from teach.logger import create_logger

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
        "--inference_output_dir",
        type=str,
        required=True,
        help="Directory containing output files from running inference on EDH instances",
    )
    arg_parser.add_argument(
        "--split",
        type=str,
        default="valid_seen",
        choices=["train", "valid_seen", "valid_unseen", "test_seen", "test_unseen"],
        help="One of train, valid_seen, valid_unseen, test_seen, test_unseen",
    )
    arg_parser.add_argument(
        "--max_init_tries",
        type=int,
        default=5,
        help="Max attempts to correctly initialize an instance before declaring it as a failure",
    )
    arg_parser.add_argument("--max_traj_steps", type=int, default=1000, help="Max predicted trajectory steps")
    arg_parser.add_argument("--max_api_fails", type=int, default=30, help="Max allowed API failures")
    arg_parser.add_argument("--metrics_file", type=str, required=True, help="File used to store metrics")
    args = arg_parser.parse_args()

    edh_instance_files = set(os.listdir(os.path.join(args.data_dir, "edh_instances", args.split)))
    output_files = [
        os.path.join(args.inference_output_dir, f)
        for f in os.listdir(args.inference_output_dir)
        if re.sub("inference__", "", f) in edh_instance_files
    ]
    pred_action_files = [re.sub("inference__", "pred_actions__", f) for f in output_files]

    edh_instance_files_missing_output = list(
        set(edh_instance_files).difference([os.path.basename(f).split("__")[1] for f in output_files])
    )
    logger.info("Evaluating split %s requiring %d files" % (args.split, len(edh_instance_files)))
    logger.info(
        "Found output files for %d instances; treating remaining %d as failed..."
        % (len(output_files), len(edh_instance_files_missing_output))
    )

    traj_stats = dict()
    for idx, output_file in enumerate(output_files):
        pred_actions_file = pred_action_files[idx]
        if not os.path.isfile(pred_actions_file):
            logger.warning(
                "Skipping EDH instance %s with output file %s due to missing predicted actions file %s"
                % (os.path.basename(output_file).split("__")[1], output_file, pred_actions_file)
            )
            edh_instance_files_missing_output.append(os.path.basename(output_file).split("__")[1])
        instance_id, traj_metrics = load_traj_metrics(output_file, pred_actions_file, args)
        traj_stats[instance_id] = traj_metrics

    for instance_file in edh_instance_files_missing_output:
        instance_id = re.sub(".json", "", os.path.basename(instance_file))
        game_id = instance_id.split(".")[0]
        traj_metrics = create_new_traj_metrics({"instance_id": instance_id, "game_id": game_id})
        traj_metrics["error"] = 1

    results = aggregate_metrics(traj_stats, args)
    print("-------------")
    print(
        "SR: %d/%d = %.3f"
        % (results["success"]["num_successes"], results["success"]["num_evals"], results["success"]["success_rate"])
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

    results["traj_stats"] = traj_stats
    with open(args.metrics_file, "w") as h:
        json.dump(results, h)


if __name__ == "__main__":
    main()
