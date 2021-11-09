# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import json
import os
import sys
import traceback
from multiprocessing import Process

from teach.logger import create_logger
from teach.replay.episode_replay import EpisodeReplay

logger = create_logger(__name__)

# playback status keys
CNT_PLAYED_BACK = "n_games_played_back"
CNT_API_SUCC = "n_games_api_succ"
CNT_TASK_SUCC = "n_games_task_succ"
PLAYBACK_FAILURES = "playback_failures"


class EpisodeReplayWorker(Process):
    def __init__(self, idx, cmd_args, **kwargs):
        """
        Initialize a Process worker to run a set of games
        idx: index of the process
        cmd_args: arguments passed into main script
        **kwargs: arguments to track process status
        """
        super(Process, self).__init__()
        self.idx = idx
        self.cmd_args = cmd_args
        self.kwargs = kwargs

    def run(self):
        """
        Looping through games and write out status at the end
        """
        for game_fn in self.kwargs.get("files_to_play"):
            self.replay_game(game_fn)
        logger.info(
            "In process %d, playback failures %d"
            % (self.idx, len(self.kwargs.get("playback_status")[PLAYBACK_FAILURES]))
        )
        self.write_out_status()

    def replay_game(self, game_fn):
        """
        replay individual game
        """
        cmd_args = self.cmd_args
        replay_status = self.kwargs.get("replay_status")
        logger.info("Processing file %s" % game_fn)
        replay_status[game_fn] = dict()
        try:
            er = EpisodeReplay("thor", ["ego", "allo", "targetobject"])
            er.set_episode_by_fn_and_idx(game_fn, cmd_args.task_idx, cmd_args.episode_idx)
            if cmd_args.write_frames_dir:
                frame_dir = os.path.join(cmd_args.write_frames_dir, game_fn.split("/")[-1].split(".")[0])
            else:
                frame_dir = None
            api_succ, task_succ = er.play_episode(
                obs_dir=frame_dir,
                realtime=cmd_args.realtime,
                write_frames=cmd_args.write_frames,
                write_states=cmd_args.write_states,
                write_episode_progress=cmd_args.write_episode_progress,
                turn_on_lights=cmd_args.turn_on_lights,
                force_replay=cmd_args.force_replay,
            )
            replay_status[game_fn]["replay_ran"] = 1
            replay_status[game_fn]["api_success"] = int(api_succ)
            replay_status[game_fn]["task_success"] = int(task_succ)
            self.update_playback_status(api_succ, task_succ, game_fn)
            if cmd_args.create_video:
                er.stitch_episode_video(frame_dir, cmd_args.font_fn)
            self.write_out_status()
        except (KeyboardInterrupt, SystemExit):
            self.update_error_status(game_fn, "interrupt")
            raise
        except Exception:
            self.update_error_status(game_fn, "exec_error")
            traceback.print_exc(file=sys.stdout)
            raise

    def update_playback_status(self, api_succ, task_succ, game_fn):
        """
        update overall counts
        """
        playback_status = self.kwargs.get("playback_status")
        playback_status[CNT_PLAYED_BACK] += 1
        if api_succ:
            playback_status[CNT_API_SUCC] += 1
        if task_succ:
            playback_status[CNT_TASK_SUCC] += 1
        if not api_succ or not task_succ:
            playback_status[PLAYBACK_FAILURES].append(game_fn)
        logger.info(
            "api success %d/%d; task success %d/%d"
            % (
                playback_status[CNT_API_SUCC],
                playback_status[CNT_PLAYED_BACK],
                playback_status[CNT_TASK_SUCC],
                playback_status[CNT_PLAYED_BACK],
            )
        )

    def write_out_status(self):
        """
        write out status file for this process
        """
        replay_status = self.kwargs.get("replay_status")
        status_file = self.kwargs.get("status_file")
        with open(status_file, "w") as h:
            json.dump(replay_status, h)

    def update_error_status(self, game_fn, error):
        """
        update replay status with error and write out status file
        """
        replay_status = self.kwargs.get("replay_status")
        status_file = self.kwargs.get("status_file")
        replay_status[game_fn]["replay_ran"] = 1
        replay_status[game_fn][error] = 1
        replay_status[game_fn]["api_success"] = 0
        replay_status[game_fn]["task_success"] = 0
        with open(status_file, "w") as h:
            json.dump(replay_status, h)
