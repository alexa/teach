#!/usr/bin/env python

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
import json
import os
import re
import time

from teach.logger import create_logger
from teach.replay.episode_replay_worker import EpisodeReplayWorker

logger = create_logger(__name__)

# playback status keys
CNT_PLAYED_BACK = "n_games_played_back"
CNT_API_SUCC = "n_games_api_succ"
CNT_TASK_SUCC = "n_games_task_succ"
PLAYBACK_FAILURES = "playback_failures"


def init_playback_status():
    """
    initialize overall counts to 0
    """
    playback_status = {CNT_PLAYED_BACK: 0, CNT_API_SUCC: 0, CNT_TASK_SUCC: 0, PLAYBACK_FAILURES: []}
    return playback_status


def combine_process_status_files(cmd_args, replay_status, status_files):
    """
    combine status file for each process into one final status file
    """
    for status_file in status_files:
        if os.path.isfile(status_file):
            with open(status_file) as h:
                thread_replay_status = json.load(h)
            replay_status.update(thread_replay_status)
            with open(cmd_args.status_out_fn, "w") as h:
                json.dump(replay_status, h)


def load_preran_status(cmd_args, game_fns):
    """
    load pre ran status and check any mismatches
    """
    replay_status = dict()
    playback_status = dict()
    with open(cmd_args.status_out_fn) as h:
        replay_status = json.load(h)
        playback_status[CNT_PLAYED_BACK] = len([k for k, v in replay_status.items() if v["replay_ran"] == 1])
        playback_status[CNT_API_SUCC] = len([k for k, v in replay_status.items() if v["api_success"] == 1])
        playback_status[CNT_TASK_SUCC] = len([k for k, v in replay_status.items() if v["task_success"] == 1])
        playback_status[PLAYBACK_FAILURES] = [k for k, v in replay_status.items() if v["replay_ran"] == 0]
    replayed_codes = [k.split("/")[-1].split(".")[0] for k, v in replay_status.items() if v["replay_ran"] == 1]
    replayed_folders = set(os.listdir(cmd_args.write_frames_dir))
    logs_without_folders = set(replayed_codes).difference(replayed_folders)
    folders_without_logs = set(replayed_folders).difference(replayed_codes)
    logger.info("Replayed codes without folder = %s" % ",".join(logs_without_folders))
    logger.info("Folders without replay stat = %s" % ",".join(folders_without_logs))
    logger.info("# Replayed codes without folder = %d" % len(logs_without_folders))
    logger.info("# Folders without replay stat = %d" % len(folders_without_logs))
    if len(logs_without_folders) > 0 or len(folders_without_logs) > 1:
        raise RuntimeError(
            "Mismatch between status file and output images folder: "
            + "\nReplayed codes without folder = "
            + str(logs_without_folders)
            + "\nFolders without replay stat = "
            + str(folders_without_logs)
            + "\n# Replayed codes without folder = "
            + str(len(logs_without_folders))
            + "\n# Folders without replay stat = "
            + str(len(folders_without_logs))
        )
    elif len(folders_without_logs) == 1:
        dangling_folder = list(folders_without_logs)[0]
        game_file_for_dangling_folder = [f for f in game_fns if f.split("/")[-1].split(".")[0] == dangling_folder]
        if len(game_file_for_dangling_folder) != 1:
            raise RuntimeError(
                "Trying to handle dangling folder "
                + str(folders_without_logs)
                + ", expected 1 matching game file but found: "
                + str(game_file_for_dangling_folder)
            )
        replay_status[game_file_for_dangling_folder[0]] = dict()
        replay_status[game_file_for_dangling_folder[0]]["replay_ran"] = 1
        replay_status[game_file_for_dangling_folder[0]]["api_success"] = 0
        replay_status[game_file_for_dangling_folder[0]]["task_success"] = 0
    files_to_play = [f for f in game_fns if f.split("/")[-1].split(".")[0] not in replayed_folders]
    return files_to_play, playback_status, replay_status


def get_game_file_names(cmd_args):
    """
    gather all game files from game_dir
    """
    game_fns = [cmd_args.game_fn] if cmd_args.game_fn is not None else []
    logger.info("game_fns = %s" % game_fns)
    if cmd_args.game_dir is not None:
        logger.info("game_dir = %s" % cmd_args.game_dir)
        for root, _, fns in os.walk(cmd_args.game_dir):
            for fn in fns:
                if "game" in fn.split("."):
                    game_fns.append(os.path.join(root, fn))
    return game_fns


def process_arguments():
    parser = argparse.ArgumentParser(description="Read in a game log, replay an episode, and create a video.")
    parser.add_argument("--game_fn", type=str, required=False, default=None, help="The game logfile to read")
    parser.add_argument(
        "--game_dir",
        type=str,
        required=False,
        default=None,
        help="The directory to read all game files from",
    )
    parser.add_argument("--task_idx", type=int, default=0, help="The task index to replay in the game file")
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=0,
        help="The episode index to replay in the task metadata",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Whether to play the episode back in real time by waiting between actions",
    )
    parser.add_argument(
        "--write_states",
        action="store_true",
        help="Whether to store states at every time step",
    )
    parser.add_argument(
        "--write_frames_dir",
        type=str,
        required=False,
        help="Directory to write frames for video creation; won't write frames if not specified",
    )
    parser.add_argument(
        "--write_episode_progress",
        action="store_true",
        help="Whether to store episode progress at every time step",
    )
    parser.add_argument(
        "--write_frames",
        action="store_true",
        help="Whether to write frames at every time step",
    )
    parser.add_argument(
        "--turn_on_lights",
        action="store_true",
        help="Specify this to turn on the lights even if the game had them off",
    )
    parser.add_argument(
        "--force_replay",
        action="store_true",
        help="Specify this to force replay the game even if the game has been replayed before",
    )
    parser.add_argument("--create_video", action="store_true", help="Whether to write a video into the frames dir")
    parser.add_argument(
        "--font_fn",
        type=str,
        default="/System/Library/Fonts/SFNSMono.ttf",
        help="Path to a font file to be used during video stitch.",
    )
    parser.add_argument(
        "--status_out_fn",
        type=str,
        default="replay_stats.json",
        help="Path to a json file, stores whether replay ran, and API and task success during replay.",
    )
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use")
    args = parser.parse_args()
    if args.game_fn is None and args.game_dir is None:
        print("please specify either game_fn or game_dir")
        exit(1)
    if args.game_fn is not None and args.game_dir is not None:
        print("you can't have both game_fn and game_dir specified")
        exit(1)
    if not args.create_video and args.write_frames_dir is None:
        print("please specify either create_video or write_frames_dir or both")
        exit(1)
    return args


def main():
    cmd_args = process_arguments()
    game_fns = get_game_file_names(cmd_args)
    playback_status = init_playback_status()
    replay_status = dict()

    files_to_play = game_fns
    if os.path.isfile(cmd_args.status_out_fn) and cmd_args.write_frames_dir:
        files_to_play, playback_status, replay_status = load_preran_status(cmd_args, game_fns)

    logger.info("len(files_to_play) = %d" % len(files_to_play))

    if cmd_args.num_processes == 1:
        worker = EpisodeReplayWorker(
            1,
            cmd_args,
            files_to_play=files_to_play,
            playback_status=playback_status,
            replay_status=replay_status,
            status_file=cmd_args.status_out_fn,
        )
        worker.run()
    else:
        num_files_per_process = int(len(files_to_play) / cmd_args.num_processes) + 1
        processes = list()
        status_files = list()
        try:
            for idx in range(cmd_args.num_processes):
                games_start_idx = idx * num_files_per_process
                games_end_idx = min(games_start_idx + num_files_per_process, len(files_to_play))
                files_for_process = files_to_play[games_start_idx:games_end_idx]
                status_file = re.sub(".json", "_" + str(idx) + ".json", cmd_args.status_out_fn)
                worker = EpisodeReplayWorker(
                    idx,
                    cmd_args,
                    files_to_play=files_for_process,
                    playback_status=playback_status,
                    replay_status=replay_status,
                    status_file=status_file,
                )
                worker.start()
                processes.append(worker)
                status_files.append(status_file)
                time.sleep(0.1)
        finally:
            for worker in processes:
                worker.join()

            combine_process_status_files(cmd_args, replay_status, status_files)


if __name__ == "__main__":
    main()
