# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import os

import numpy as np

from teach.utils import create_task_thor_from_state_diff


def evaluate_traj(success, edh_instance, traj_len, final_gc_total, final_gc_satisfied):
    init_gc_satisfied = min(
        edh_instance["expected_init_goal_conditions_total"], edh_instance["expected_init_goal_conditions_satisfied"]
    )
    final_gc_satisfied = min(final_gc_total, final_gc_satisfied)
    goal_condition_success_rate = 1.0 - (
        (final_gc_total - final_gc_satisfied)
        / (edh_instance["expected_init_goal_conditions_total"] - init_gc_satisfied)
    )

    # SPL
    gt_path_len = len(edh_instance["driver_actions_future"])
    s_spl = (1 if success else 0) * min(1.0, gt_path_len / float(max(traj_len, gt_path_len)))
    pc_spl = goal_condition_success_rate * min(1.0, gt_path_len / float(max(traj_len, gt_path_len)))
    # path length weighted SPL
    plw_s_spl = s_spl * gt_path_len
    plw_pc_spl = pc_spl * gt_path_len
    return {
        "completed_goal_conditions": int((edh_instance["expected_init_goal_conditions_total"] - init_gc_satisfied))
        - int(final_gc_total - final_gc_satisfied),
        "total_goal_conditions": int(edh_instance["expected_init_goal_conditions_total"] - init_gc_satisfied),
        "goal_condition_success": float(goal_condition_success_rate),
        "success_spl": float(s_spl),
        "path_len_weighted_success_spl": float(plw_s_spl),
        "goal_condition_spl": float(pc_spl),
        "path_len_weighted_goal_condition_spl": float(plw_pc_spl),
        "path_len_weight": int(gt_path_len),
        "success": int(success),
        "traj_len": int(traj_len),
    }


def create_new_traj_metrics(edh_instance):
    return {
        "instance_id": edh_instance["instance_id"],
        "game_id": edh_instance["game_id"],
        "completed_goal_conditions": 0,
        "total_goal_conditions": 0,
        "goal_condition_success": 0.0,
        "success_spl": 0.0,
        "path_len_weighted_success_spl": 0.0,
        "goal_condition_spl": 0.0,
        "path_len_weighted_goal_condition_spl": 0.0,
        "gt_path_len": 0,
        "reward": 0.0,
        "success": 0,
        "traj_len": 0,
        "predicted_stop": 0,
        "num_api_fails": 0,
        "error": 0,
        "init_success": 0,
        "pred_actions": [],
    }


def aggregate_metrics(traj_stats, args):
    """
    compute overall success and goal_condition success rates along with path-weighted metrics
    """
    # stats
    num_successes = len([k for k, v in traj_stats.items() if v["success"] == 1])
    num_evals = len(traj_stats.keys())
    total_path_len_weight = sum([v["gt_path_len"] for k, v in traj_stats.items()])
    completed_goal_conditions = sum([v["completed_goal_conditions"] for k, v in traj_stats.items()])
    total_goal_conditions = sum([v["total_goal_conditions"] for k, v in traj_stats.items()])
    num_predicted_stops = sum([v["predicted_stop"] for k, v in traj_stats.items()])
    num_fails_by_api_limit = sum([v["num_api_fails"] >= args.max_api_fails for k, v in traj_stats.items()])
    num_fails_by_traj_len_limit = sum([v["traj_len"] >= args.max_traj_steps for k, v in traj_stats.items()])
    num_fails_by_error = sum([v["error"] for k, v in traj_stats.items()])

    # metrics
    sr = float(num_successes) / num_evals
    pc = completed_goal_conditions / float(total_goal_conditions)
    if total_path_len_weight > 0.0 and not np.isclose(total_path_len_weight, 0.0):
        plw_sr = float(sum([v["path_len_weighted_success_spl"] for k, v in traj_stats.items()])) / total_path_len_weight
        plw_pc = (
            float(sum([v["path_len_weighted_goal_condition_spl"] for k, v in traj_stats.items()]))
            / total_path_len_weight
        )
    else:
        plw_sr = plw_pc = 0.0

    # result table
    res = dict()
    res["success"] = {"num_successes": num_successes, "num_evals": num_evals, "success_rate": sr}
    res["goal_condition_success"] = {
        "completed_goal_conditions": completed_goal_conditions,
        "total_goal_conditions": total_goal_conditions,
        "goal_condition_success_rate": pc,
    }
    res["path_length_weighted_success_rate"] = plw_sr
    res["path_length_weighted_goal_condition_success_rate"] = plw_pc
    res["num_predicted_stops"] = num_predicted_stops
    res["num_fails_by_api_limit"] = num_fails_by_api_limit
    res["num_fails_by_traj_len_limit"] = num_fails_by_traj_len_limit
    res["num_fails_by_error"] = num_fails_by_error

    return res


def load_traj_metrics(output_file, pred_actions_file, args):
    with open(output_file) as h:
        game_json = json.load(h)
    edh_instance_file = os.path.join(
        args.data_dir, "edh_instances", args.split, os.path.basename(output_file).split("__")[1]
    )
    with open(edh_instance_file) as h:
        edh_instance = json.load(h)

    with open(pred_actions_file) as h:
        pred_actions = json.load(h)

    edh_check_task = create_task_thor_from_state_diff(edh_instance["state_changes"])
    final_state_objects = game_json["tasks"][0]["episodes"][0]["final_state"]["objects"]
    final_state_custom_metadata = game_json["tasks"][0]["episodes"][0]["final_state"]["custom_object_metadata"]
    for obj in final_state_objects:
        if obj["objectId"] in final_state_custom_metadata:
            obj.update(final_state_custom_metadata[obj["objectId"]])
    progress_check_output = edh_check_task.check_episode_progress(final_state_objects)
    success = progress_check_output["success"]
    final_goal_conditions_total = progress_check_output["goal_conditions_total"]
    final_goal_conditions_satisfied = progress_check_output["goal_conditions_satisfied"]

    traj_metrics = create_new_traj_metrics(edh_instance)
    traj_metrics["game_id"] = edh_instance["game_id"]
    traj_metrics["instance_id"] = edh_instance["instance_id"]
    traj_metrics.update(
        evaluate_traj(
            success, edh_instance, len(pred_actions), final_goal_conditions_total, final_goal_conditions_satisfied
        )
    )

    return edh_instance["instance_id"], traj_metrics
