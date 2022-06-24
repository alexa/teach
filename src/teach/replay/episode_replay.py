# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import copy
import glob
import json
import os
import pickle
import time

import numpy as np
import tqdm
from PIL import Image, ImageDraw, ImageFont

from teach.dataset.dataset import Dataset
from teach.dataset.definitions import Definitions
from teach.logger import create_logger
from teach.simulators import simulator_factory
from teach.utils import get_state_changes, reduce_float_precision

logger = create_logger(__name__)
definitions = Definitions(version="2.0")
action_id_to_info = definitions.map_actions_id2info


class EpisodeReplay:
    def __init__(self, simulator_name, live_feeds):
        """
        Initialize a simulator to use.
        live_feeds: list of string names for camera feeds whose data we should save during replay
        simulator_name: name of simulator that is registered in simulator_factory
        """
        self.simulator = simulator_factory.factory.create(simulator_name, web_window_size=900)
        self.simulator_name = simulator_name

        for name_feed in live_feeds:
            if name_feed not in self.simulator.live_feeds:
                self.simulator.live_feeds.add(name_feed)

        # State data.
        self.task = self.task_params = self.episode = None

    def set_episode_by_fn_and_idx(self, game_fn, task_idx, episode_idx):
        """
        Read in an episode from file.
        game_fn: the game logfile to read
        task_idx: the task in the game logfile to read
        episode_idx: the episode in the task metadata to read
        """

        structured_log = Dataset.import_json(file_name=game_fn, process_init_state=False, version="2.0")
        task = structured_log.tasks[task_idx]
        self.task = task.task_name
        self.task_params = task.task_params
        self.episode = task.episodes[episode_idx]

    def play_episode(
        self,
        obs_dir=None,
        realtime=False,
        force_replay=False,
        write_frames=False,
        write_states=False,
        write_episode_progress=False,
        turn_on_lights=False,
        task=None,
        shutdown_on_finish=True,
    ):
        """
        Play back the interactions in an episode.
        obs_dir: the directory to write observation files; if None, skips writing raw observation data.
        realtime: if True, play back episode with delays between actions based on user times.
        force_replay: if False, skips playback if the obs_dir is non-emtpy.
        write_frames: if True, frames will be written out at every time step.
        write_states: if True, states will be written out at every time step.
        write_episode_progress: if True, episode progress will be written out at every time step.
        turn_on_lights: if True, will turn on the lights even if the game had them off.
        """
        if not force_replay and obs_dir is not None and os.path.isdir(obs_dir) and len(os.listdir(obs_dir)) > 0:
            logger.warn("play_episode skipping playback in non-empty dir '%s'" % obs_dir)
            return False, False

        api_success, init_state = self.set_up_new_episode(
            obs_dir, turn_on_lights, task
        )
        init_state_objects = self.simulator.get_objects()

        target_object_active = False
        for idx in range(len(self.episode.interactions)):
            api_success, target_object_active = self._play_single_interaction(
                api_success,
                idx,
                init_state,
                obs_dir,
                realtime,
                target_object_active,
                write_episode_progress,
                write_frames,
                write_states,
            )
        self._write_last_states_and_frames(init_state, obs_dir, target_object_active, write_frames, write_states)

        _, task_success, _, _, _ = self.simulator.check_episode_progress(self.simulator.current_task)

        if shutdown_on_finish:
            self.simulator.shutdown_simulator()
            logger.info(
                "Episode ended, took %d steps; api success=%d; task success=%d"
                % (len(self.episode.interactions), int(api_success), int(task_success))
            )

        return api_success, task_success

    def stitch_episode_video(self, obs_dir, font_fn, force_replay=False):
        """
        Stitch together a video of the episode for demo/inspection purposes.
        obs_dir - the observations to read in
        force_replay: if False, skips playback if the [obs_dir]/video is non-emtpy.
        """
        out_dir = os.path.join(obs_dir, "video")
        if not force_replay and os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
            logger.warn("stitch_episode_video skipping video stitch in non-empty dir '%s'" % out_dir)
            return

        # Assemble frames to be stitched together and determine order.
        frame_fns = glob.glob(os.path.join(obs_dir, "*.j*"))  # 'jpeg', 'json'
        timestamps_to_fns = {}
        for fn in frame_fns:
            t = ".".join(fn.split(".")[2:-1])
            if t not in timestamps_to_fns:
                timestamps_to_fns[t] = []
            timestamps_to_fns[t].append(fn)

        # Create the frames of the video. They are a 4 panel showing driver, commander, target object, and
        # target object seg mask at each timestep.
        frame_border = 50  # pixels
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        frame = ref_im = None
        for fn in frame_fns:
            if fn.split(".")[-1] == "jpeg":
                ref_im = Image.open(fn)
                frame = Image.new(
                    "RGB",
                    (frame_border * 2 + ref_im.width * 4, frame_border * 2 + ref_im.height * 2),
                )
                break
        frame_layout = {
            "driver.frame": (0, 0),
            "commander.frame": (1, 0),
            "targetobject.frame": (0, 1),
            "targetobject.mask": (1, 1),
            "text": (2, 0),
            "bottomright": (4, 2),
        }

        frame_idx = 0
        logger.info("Iterating through frames to assemble video tiles...")
        for t in tqdm.tqdm(sorted(timestamps_to_fns, key=lambda x: float(x) if x and x != "end" else float("inf"))):
            for fn in timestamps_to_fns[t]:
                frame_type = ".".join(fn.split("/")[-1].split(".")[:2])
                if frame_type in frame_layout:  # visual observation
                    im = Image.open(fn)
                    frame.paste(
                        im,
                        (
                            frame_border + ref_im.width * frame_layout[frame_type][0],
                            frame_border + ref_im.height * frame_layout[frame_type][1],
                        ),
                    )
                else:  # text observation
                    with open(fn, "r") as f:
                        contents = json.load(f)
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle(
                        (
                            (
                                frame_border + ref_im.width * frame_layout["text"][0],
                                frame_border + ref_im.height * frame_layout["text"][1],
                            ),
                            (
                                frame_border + ref_im.width * frame_layout["bottomright"][0],
                                frame_border + ref_im.height * frame_layout["bottomright"][1],
                            ),
                        ),
                        fill="black",
                    )
                    font_size = 64
                    font = ImageFont.truetype(font_fn, font_size)
                    s = "%s: %s" % (frame_type, json.dumps(contents))
                    line_idx = 0
                    chars_per_line = 44
                    while len(s) > 0:
                        draw.text(
                            (
                                frame_border + ref_im.width * frame_layout["text"][0] + 5,
                                frame_border + ref_im.height * frame_layout["text"][1] + (font_size + 5) * line_idx,
                            ),
                            s[: min(len(s), chars_per_line)],
                            (255, 255, 255),
                            font=font,
                        )
                        s = s[min(len(s), chars_per_line) :]
                        line_idx += 1
            frame.save(os.path.join(out_dir, "%05d.jpeg" % frame_idx), format="jpeg")
            frame_idx += 1
        logger.info("... done; wrote %d assembled frames" % frame_idx)

        if frame_idx > 0:
            cmd = (
                'ffmpeg -r 1 -start_number 0 -i "'
                + out_dir
                + '/%05d.jpeg" -c:v libx264 -vf "fps=25,format=yuv420p" '
                + os.path.join(out_dir, "video.mp4")
            )
            logger.info("Executing: ", cmd)
            os.system(cmd)
            logger.info("... done")
        else:
            logger.warn("no frames extracted to stich video for %s" % out_dir)

    def write_progress(self, frame_idx, obs_dir):
        progress_check_output = self.simulator.current_task.check_episode_progress(
            self.simulator.get_objects(self.simulator.controller.last_event), self.simulator
        )
        with open(
            os.path.join(
                obs_dir,
                "progress_check_output.%s.pkl" % (frame_idx,),
            ),
            "wb",
        ) as f:
            pickle.dump(progress_check_output, f)

    def write_cur_state(self, frame_idx, obs_dir, init_state):
        cur_state = reduce_float_precision(self.simulator.get_current_state().to_dict())
        state_diff = get_state_changes(init_state, cur_state)
        with open(
            os.path.join(
                obs_dir,
                "statediff.%s.json" % (frame_idx,),
            ),
            "w",
        ) as f:
            json.dump(state_diff, f)

    def write_frames(self, frame_idx, obs_dir, target_object_active):
        frames = self.simulator.get_latest_images()
        self._write_frame(
            frames["ego"],
            os.path.join(
                obs_dir,
                "driver.frame.%s.jpeg" % frame_idx,
            ),
        )
        self._write_frame(
            frames["allo"],
            os.path.join(
                obs_dir,
                "commander.frame.%s.jpeg" % frame_idx,
            ),
        )
        self._write_frame(
            frames["targetobject"] if target_object_active else np.zeros_like(frames["targetobject"]),
            os.path.join(
                obs_dir,
                "targetobject.frame.%s.jpeg" % frame_idx,
            ),
        )

    def _play_single_interaction(
        self,
        api_success,
        idx,
        init_state,
        obs_dir,
        realtime,
        target_object_active,
        write_episode_progress,
        write_frames,
        write_states,
    ):

        frame_idx = str(self.episode.interactions[idx].time_start)

        if obs_dir is not None and write_states:
            self.write_cur_state(frame_idx, obs_dir, init_state)

        if obs_dir is not None and write_frames:
            self.write_frames(frame_idx, obs_dir, target_object_active)

        if realtime:
            self._wait_for_real_time(idx)

        action_definition = action_id_to_info[self.episode.interactions[idx].action.action_id]
        logger.debug("taking action <<%s, %s>>" % (action_definition["action_type"], action_definition["action_name"]))
        logged_success = self.episode.interactions[idx].action.success
        interact_oid = (
            self.episode.interactions[idx].action.oid
            if action_definition["action_type"] == "ObjectInteraction"
            else None
        )
        if logged_success == 1:
            self._add_interaction(idx, interact_oid, logged_success)

            api_success = api_success & (1 == self.episode.interactions[idx].action.success)

            if obs_dir is not None and write_frames:
                if action_definition["action_type"] == "Keyboard":
                    self._write_keyboard_frame(idx, obs_dir)

            if obs_dir is not None and write_frames:
                if action_definition["action_type"] == "ProgressCheck":
                    target_object_active = self._write_progress_check(
                        idx, obs_dir, action_definition, target_object_active
                    )
        if obs_dir is not None and write_episode_progress:
            self.write_progress(frame_idx, obs_dir)
        return api_success, target_object_active

    def _add_interaction(self, idx, interact_oid, logged_success):
        self.simulator.add_interaction(self.episode.interactions[idx])
        if self.episode.interactions[idx].action.success != logged_success:
            logger.debug(
                "... action success logged %d != %d of action just taken"
                % (logged_success, self.episode.interactions[idx].action.success)
            )
        # If oid was interacted with in orig, but action failed on replay or the wrong oid was interacted when using
        # the (x, y) coords given (e.g., because objects can jitter with PhysX but shouldn't be
        # out of frame in most cases), try to just get a correct oid click directly.
        if interact_oid is not None and (
            (logged_success and not self.episode.interactions[idx].action.success)
            or interact_oid != self.episode.interactions[idx].action.oid
        ):
            # Next, override the provided user (x, y) with a randomly selected mask point (x*, y*) on the object.
            mask_frame = self.simulator.get_target_object_seg_mask(interact_oid)["mask"]
            mask_points = np.where(mask_frame == 1)[:2]
            if len(mask_points[0]) > 0:  # if any part of the object is visible in the frame, pick a point on it
                rpoint_idx = np.random.randint(0, len(mask_points[0]))
                override_interaction = copy.deepcopy(self.episode.interactions[idx])
                override_interaction.action.x = mask_points[1][rpoint_idx] / self.simulator.web_window_size
                override_interaction.action.y = mask_points[0][rpoint_idx] / self.simulator.web_window_size
                logger.info("... override interaction %s" % override_interaction)  # DEBUG
                self.simulator.add_interaction(override_interaction)
                if self.episode.interactions[idx].action.success != logged_success:
                    logger.info(
                        "...... action success logged %d != %d of override action with oid %s just taken"
                        % (
                            logged_success,
                            self.episode.interactions[idx].action.success,
                            interact_oid,
                        )
                    )
            else:  # Really nasty, object isn't even visible. Try to take the action directly.
                logger.info("... override interaction with oid %s" % interact_oid)
                cur_objs = self.simulator.get_objects()
                logger.info(
                    "Cur objs (ID, visible): %s " % str([(obj["objectId"], obj["visible"]) for obj in cur_objs])
                )
                self.simulator.add_interaction(self.episode.interactions[idx], on_oid=interact_oid, force=True)
                if self.episode.interactions[idx].action.success != logged_success:
                    logger.info(
                        "..... action success logged %d != %d of override action just taken"
                        % (logged_success, self.episode.interactions[idx].action.success)
                    )

    def set_up_new_episode(self, obs_dir=None, turn_on_lights=False, task=None):
        api_success = True
        self.simulator.reset_stored_data()
        logger.info("Starting episode...")
        self.simulator.start_new_episode(
            world=self.episode.world,
            world_type=self.episode.world_type,
            commander_embodied=True if self.episode.commander_embodied == "True" else False,
        )
        logger.info("... done")

        logger.info("Loading initial scene state...")
        _, s = self.simulator.load_scene_state(init_state=self.episode.initial_state)
        api_success = api_success & s
        logger.info("... done")

        if task is not None:
            logger.info("Setting to custom task %s" % task)
            self.simulator.set_task(task=task)
        else:
            logger.info("Setting task %s with task_params %s..." % (self.task, self.task_params))
            self.simulator.set_task_by_name(task_name=self.task, task_params=self.task_params)
        logger.info("... done")

        if turn_on_lights:
            self._turn_on_lights()

        init_state = reduce_float_precision(self.simulator.get_current_state().to_dict())

        if obs_dir is not None and not os.path.isdir(obs_dir):
            os.makedirs(obs_dir)

        return api_success, init_state

    def _write_last_states_and_frames(self, init_state, obs_dir, target_object_active, write_frames, write_states):
        frame_idx = "end"
        if obs_dir is not None and write_states:
            self.write_cur_state(frame_idx, obs_dir, init_state)

        if obs_dir is not None and write_frames:
            self.write_frames(frame_idx, obs_dir, target_object_active)

    def _write_progress_check(self, idx, obs_dir, action_definition, target_object_active):
        frames = self.simulator.get_latest_images()
        r = self.simulator.apply_progress_check(
            action_definition["action_name"],
            self.episode.interactions[idx].agent_id,
            self.episode.interactions[idx].action.query,
        )

        # If this was a progress check query, write the observation to file.
        if action_definition["action_name"] == "OpenProgressCheck":
            with open(
                os.path.join(
                    obs_dir,
                    "progresscheck.status.%s.json" % str(self.episode.interactions[idx].time_start),
                ),
                "w",
            ) as f:
                json.dump(r, f)  # Success, subgoal success, string descriptions, problem objects
        # Else, this is an oid find or object search that takes a string as input and outputs
        # a segmentation mask + actives the target object camera
        else:
            if not target_object_active:  # write target object frame if we missed it
                frames = self.simulator.get_latest_images()
                self._write_frame(
                    frames["targetobject"],
                    os.path.join(
                        obs_dir,
                        "targetobject.frame.%s.jpeg" % str(self.episode.interactions[idx].time_start),
                    ),
                )
            target_object_active = True
            with open(
                os.path.join(
                    obs_dir,
                    "progresscheck.%s.%s.json"
                    % (
                        action_definition["action_name"],
                        str(self.episode.interactions[idx].time_start),
                    ),
                ),
                "w",
            ) as f:
                # r contains success, oid shown, view pos/rot for topdown map, hotspots info.
                # In the video, we'll show whether the search was a success, what was searched for,
                # and what oid was shown.
                to_report = {
                    "success": r["success"],
                    "query": self.episode.interactions[idx].action.query,
                }
                if "shown_oid" in r:
                    to_report["shown_oid"] = r["shown_oid"]
                json.dump(to_report, f)

            # Write targetobject segmentation frame that contains exact segmentation mask for object.
            mask_frame = np.zeros_like(frames["targetobject"])
            if "shown_oid" in r and len(r["shown_oid"]) > 0:
                mask_points = np.where(self.simulator.get_target_object_seg_mask(r["shown_oid"])["mask"] == 1)[:2]
                for p in mask_points:
                    mask_frame[p] = (255, 255, 255)
            self._write_frame(
                mask_frame,
                os.path.join(
                    obs_dir,
                    "targetobject.mask.%s.jpeg" % str(self.episode.interactions[idx].time_start),
                ),
            )
        return target_object_active

    def _write_keyboard_frame(self, idx, obs_dir):
        with open(
            os.path.join(
                obs_dir,
                "keyboard.%d.%s.json"
                % (
                    self.episode.interactions[idx].agent_id,
                    str(self.episode.interactions[idx].time_start),
                ),
            ),
            "w",
        ) as f:
            json.dump(
                {
                    "agent_id": self.episode.interactions[idx].agent_id,
                    "utterance": self.episode.interactions[idx].action.utterance,
                },
                f,
            )

    def _wait_for_real_time(self, idx):
        twait = self.episode.interactions[idx].time_start - (
            self.episode.interactions[idx - 1].time_start if idx > 0 else 0
        )
        if twait > 0:
            logger.info("waiting %.2f seconds" % twait)
            time.sleep(twait)

    def _write_frame(self, np_frame_array, filename):
        pil_img = Image.fromarray(np_frame_array)
        pil_img.save(filename, format="jpeg")

    def _turn_on_lights(self):
        logger.warn("Turning on lights... This should not be used for experiments")
        objects = self.simulator.get_objects()
        light_switches = [obj for obj in objects if "LightSwitch" in obj["objectType"]]
        for obj in light_switches:
            action = {"action": "ToggleObjectOn", "agentId": 0, "objectId": obj["objectId"], "forceAction": True}
            self.simulator.controller.step(action)
