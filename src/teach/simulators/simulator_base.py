# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


import base64
import copy
import io
import json
import logging
import os
import time
import uuid
from datetime import datetime

import numpy as np
from PIL import Image
from pydub import AudioSegment

from teach.dataset.actions import (
    Action_Audio,
    Action_Keyboard,
    Action_MapGoal,
    Action_Motion,
    Action_ObjectInteraction,
    Action_ProgressCheck,
)
from teach.dataset.dataset import Dataset
from teach.dataset.episode import Episode
from teach.dataset.initialization import Initialization
from teach.dataset.interaction import Interaction
from teach.dataset.pose import Pose
from teach.logger import create_logger
from teach.utils import save_dict_as_json

logger = create_logger(__name__)


class SharedContent:
    def __init__(self):
        self.ego = np.random.randint(0, 255, size=(512, 512, 3), dtype=np.uint8)
        self.info = {"message": ""}


class SimulatorBase:
    """
    This class contains most of the common implementations for simulators
    """

    def __init__(
        self,
        task_type="eqa_complex",
        comments=None,
        fps=25,
        logger_name=__name__,
        logger_level=logging.DEBUG,
        dir_out=None,
        s3_bucket_name=None,
    ):
        """
        Constructor for Simulator_Base

        :param task_type: Type of task. This is currently user-defined. Default = 'eqa_complex'
        :type task_type: String

        :param comments: Informative comments for the entire data collection session. Default = None (use current day, time)
        :type comments: String

        :param fps: Maximum frame rate for video feed. Default = 25
        :type fps: Integer

        :param logger_name: Name of logger. Default = __name__ (name of the current module)
        :type logger_name: String

        :param logger_level: Level for logger. Default = logging.DEBUG
        :type logger_level: Enumeration. See logging.setLevel()

        :param dir_out: Output directory for logging
        :type dir_out: String

        :param s3_bucket_name: S3 bucket for logging
        :type s3_bucket_name: String
        """

        if comments is None:
            comments = datetime.now().strftime("%B %d, %Y: %H:%M:%S")

        self.fps = fps
        time_start = time.time()
        self._dataset = Dataset(task_type=task_type, definitions=None, comments=comments, version="2.0")
        time_end = time.time()

        self.__reset_helper(dir_out=dir_out)
        self.logger = create_logger(logger_name, level=logger_level)
        self.logger.info("Time to create dataset definitions: %s sec" % (time_end - time_start))

        self.live_feeds = set()  # set of feed names to bother encoding for emission
        self.last_images = {}

        self.current_task = self.start_time = self.current_episode = None
        self.s3_bucket_name = s3_bucket_name

    def get_task(self):
        return self.current_task

    def set_task(self, task, task_params=None, comments=""):
        raise NotImplementedError("Derived class must implement this!")

    def set_task_by_id(self, task_id: int, task_params=None, comments=""):
        raise NotImplementedError("Derived class must implement this!")

    def set_task_by_name(self, task_name: str, task_params=None, comments=""):
        raise NotImplementedError("Derived class must implement this!")

    def reset_stored_data(self):
        """
        This removes data of previous tasks / episodes from the simulator object and should be used with caution
        This should precede calls to start_new_episode() and set_task() to ensure that a future call to save() or done()
        will save session data properly.
        """
        logger.info("Resetting dataset object and removing previously stored episodes...")
        task_type = self._dataset.task_type
        comments = self._dataset.comments
        self._dataset = Dataset(task_type=task_type, definitions=None, comments=comments, version="2.0")

    def start_new_episode(
        self,
        world=None,
        world_type=None,
        object_tuples=None,
        commander_embodied=None,
        episode_id=None,
        randomize_object_search=False,
    ):
        """
        Start a new episode. Best to have diverse episodes such as different worlds, starting points,
        states. All episodes under a task should contain the same task. They may contain different
        intermediate utterances or motions.

        :param world: Identifier for the world. Default = None
        :type world: String

        :param world_type: Identifier for the world type (examples: "Kitchen", "Single-storied"). Default = None
        :type world_type: String

        :param object_tuples: List of tuples (object_type, object). (None, None) means random object_type and object. Default = None
        :type object_tuples: List of tuples made up of pairs of strings

        :param commander_embodied:
            True for commander+driver both embodied agents
            False/None for commander floating camera.
            Default = None
        :type commander_embodied: Boolean

        :param episode_id: Unique identifier for this episode; will be generated with uuid if not provided.
        :type episode_id: String

        :param randomize_object_search: This is relevant in applications that use the function set_target_object_view.
            If no object ID is specified, setting this to True will cause a random valid object to be picked every time.
        :type randomize_object_search: Bool
        """

        self.start_time = time.time()
        eid = uuid.uuid4().hex if episode_id is None else episode_id
        new_episode = Episode(eid, world, world_type, commander_embodied, initial_state=None, interactions=[])
        self.current_episode = new_episode
        self.to_broadcast["info"] = {"message": "Initial state"}

    def add_interaction(self, interaction):
        """
        Execute an Interaction - a formatted action - and add it to the current episode
        :param interaction: instance of class Interaction defined in dataset.py
        """
        if not self.is_ready:
            message = "Simulator was not initialized. Possible resolution: Start new episode."
            self.logger.error(message)
            raise Exception(message)

        if self.current_episode is None:
            message = "Not in an active episode. Start a new one."
            self.logger.warning(message)
            raise Exception(message)

        if self.is_record_mode:
            self.current_episode.add_interaction(interaction)

    def apply_motion(self, motion_name, agent_id):
        """
        Execute navigation action specified by motion_name
        :param motion_name: Action name of action defined in default_definitions.json with action_type Motion
        :param agent_id: 0 for Commander and 1 for Driver/ Follower
        """
        agent_name = self._dataset.definitions.map_agents_id2info[agent_id]["agent_name"]
        self.logger.debug("%s: %s" % (agent_name, motion_name))

        action_definition = self._dataset.definitions.map_actions_name2info.get(motion_name)
        if action_definition is None:
            self.logger.error("Unsupported motion: %s" % motion_name)
            return False, "Unsupported motion: %s" % motion_name, ""
        if "pose_delta" not in action_definition or action_definition["pose_delta"] is None:
            self.logger.error("Unsupported motion: %s" % motion_name)
            return False, "Unsupported motion: %s" % motion_name, ""

        action = Action_Motion(
            action_id=action_definition["action_id"],
            time_start=time.time() - self.start_time,
            duration=1,
            pose=Pose(0, 0, 0, 0, 0, 0),
            pose_delta=Pose.from_array(action_definition["pose_delta"]),
        )
        interaction = Interaction(agent_id=agent_id, action=action, is_object=False, status=None)
        sim_succ, err_message, help_message = self.add_interaction(interaction)
        self.to_broadcast["info"] = {"message": motion_name, "success": sim_succ, "sim_message": err_message}
        if help_message is not None:
            self.to_broadcast["info"]["help_message"] = help_message
        return sim_succ, err_message, help_message

    def apply_map_goal(self, goal_name, agent_id, start_x, start_y, end_x, end_y):
        """
        Identify and execute a series of lower-level motion actions to reach a destination.
        :param goal_name: Action name of action defined in default_definitions.json with action_type MapGoal
        :param agent_id: 0 for Commander and 1 for Driver/ Follower
        :param start_x: x-coordinate of start position on top-down map as shown in data collection interface
        :param start_y: y-coordinate of start position on top-down map as shown in data collection interface
        :param end_x: x-coordinate of desired end position on top-down map as shown in data collection interface
        :param end_y: y-coordinate of desired end position on top-down map as shown in data collection interface
        """
        agent_name = self._dataset.definitions.map_agents_id2info[agent_id]["agent_name"]
        self.logger.debug("%s: %s @ %.2f,%.2f -> %.2f,%.2f" % (agent_name, goal_name, start_x, start_y, end_x, end_y))

        action_definition = self._dataset.definitions.map_actions_name2info.get(goal_name)
        if action_definition is None:
            self.logger.error("Unsupported map goal: %s" % goal_name)
            return

        action = Action_MapGoal(
            action_id=action_definition["action_id"],
            time_start=time.time() - self.start_time,
            duration=1,
            start_x=start_x,
            start_y=start_y,
            end_x=end_x,
            end_y=end_y,
        )
        interaction = Interaction(agent_id=agent_id, action=action, is_object=False, status=None)
        sim_succ, action_sequence = self.add_interaction(interaction)
        self.to_broadcast["info"] = {
            "message": "%s: %.2f,%.2f->%.2f,%.2f" % (goal_name, start_x, start_y, end_x, end_y),
            "success": sim_succ,
            "action_sequence": action_sequence,
        }

    def apply_object_interaction(self, interaction_name, agent_id, x, y):
        """
        Execute object interaction action specified by interaction_name on object at relative coordinate (x, y) in the
        egocentric frame of the agent specified by agent_id
        :param interaction_name: Action name of action defined in default_definitions.json with action_type ObjectInteraction
        :param agent_id: 0 for Commander and 1 for Driver/ Follower
        :param x: Relative x coordinate on agent's egocentric image
        :param y: Relative y coordinate on agent's egocentric image
        """
        agent_name = self._dataset.definitions.map_agents_id2info[agent_id]["agent_name"]
        self.logger.debug("%s: %s @ %.2f,%.2f" % (agent_name, interaction_name, x, y))

        action_definition = self._dataset.definitions.map_actions_name2info.get(interaction_name)
        if action_definition is None:
            self.logger.error("Unsupported object interaction: %s" % interaction_name)
            return False, "Unsupported object interaction: %s" % interaction_name, ""

        action = Action_ObjectInteraction(
            action_id=action_definition["action_id"], time_start=time.time() - self.start_time, duration=1, x=x, y=y
        )
        interaction = Interaction(agent_id=agent_id, action=action, is_object=False, status=None)
        sim_succ, sim_msg, help_msg = self.add_interaction(interaction)
        self.to_broadcast["info"] = {
            "message": "%s: %.2f,%.2f %s" % (interaction_name, x, y, action.oid if sim_succ else ""),
            "success": sim_succ,
            "sim_message": sim_msg,
        }
        if help_msg is not None:
            self.to_broadcast["info"]["help_message"] = help_msg
        return sim_succ, sim_msg, help_msg

    def apply_progress_check(self, action_name, agent_id, query):
        """
        Execute progress check action specified by action_name. Note that if this function is used, the progress check
        action will get logged so during data collection, it is only desirable to call this when a User explicitly
        chooses to check progress. To perform an automatic progress check without logging it, use the progress_check()
        method instead.
        :param action_name: Action defined in default_definitions.json with action_type ProgressCheck
        :param agent_id: 0 for Commander and 1 for Driver/ Follower
        :param query: Specify search query for SearchObject action or object ID for SelectOid action
        """
        agent_name = self._dataset.definitions.map_agents_id2info[agent_id]["agent_name"]
        self.logger.debug('%s: %s query "%s"' % (agent_name, action_name, query))

        action_definition = self._dataset.definitions.map_actions_name2info.get(action_name)

        action = Action_ProgressCheck(
            action_id=action_definition["action_id"], time_start=time.time() - self.start_time, duration=1, query=query
        )
        interaction = Interaction(agent_id=agent_id, action=action, is_object=False, status=None)
        self.add_interaction(interaction)

        # Take appropriate action based on action name.
        if action_name == "OpenProgressCheck":
            task_desc, success, subgoals, gc_total, gc_satisfied = self.check_episode_progress(self.current_task)
            interaction.action.success = 1 if success else 0
            # Return JSON-safe encoding.
            for subgoal in subgoals:
                subgoal["success"] = 1 if subgoal["success"] else 0
                if "step_successes" in subgoal:
                    subgoal["step_successes"] = [int(v) for v in subgoal["step_successes"]]
                for step in subgoal["steps"]:
                    step["success"] = 1 if step["success"] else 0
            return {"task_desc": task_desc, "success": 1 if success else 0, "subgoals": subgoals}

        elif action_name == "SelectOid" or action_name == "SearchObject":
            obj_data = (
                self.set_target_object_view(query, None)
                if action_name == "SelectOid"
                else self.set_target_object_view(None, query)
            )
            if obj_data:
                interaction.action.success = 1
                return obj_data
            else:  # Failed to find such an object
                interaction.action.success = 0
                return {"success": False}
        else:
            raise ValueError("Unrecognized progress check action type '%s'" % action_name)

    def keyboard(self, agent_id, utterance):
        """
        Log utterances collected via chat
        :param agent_id: 0 for Commander and 1 for Driver/ Follower
        :param utterance: Utterance text
        """
        if not self.is_ready:
            message = "Simulator was not initialized. Possible resolution: Start new episode."
            self.logger.error(message)
            raise Exception(message)

        agent_name = self._dataset.definitions.map_agents_id2info[agent_id]["agent_name"]

        self.logger.debug("%s: %s" % (agent_name, utterance))

        action_definition = self._dataset.definitions.map_actions_name2info.get("Text")
        if action_definition is None:
            self.logger.error("Unsupported action: Text")
            return

        action = Action_Keyboard(
            action_id=action_definition["action_id"],
            time_start=time.time() - self.start_time,
            duration=1,
            utterance=utterance,
        )
        interaction = Interaction(agent_id=agent_id, action=action, is_object=False, status=None)
        self.add_interaction(interaction)
        self.to_broadcast["info"] = {"message": "%s: %s" % (agent_name, utterance)}

    def speech(self, agent_id, file_name, utterance):
        """
        Log utterances collected via speech
        :param agent_id: 0 for Commander and 1 for Driver/ Follower
        :param file_name: File to save audio to
        :param utterance: Audio utterance
        """
        if not self.is_ready:
            message = "Simulator was not initialized. Possible resolution: Start new episode."
            self.logger.error(message)
            raise Exception(message)

        agent_name = self._dataset.definitions.map_agents_id2info[agent_id]["agent_name"]

        self.logger.debug("%s: %s" % (agent_name, utterance))

        action_definition = self._dataset.definitions.map_actions_name2info.get("Speech")
        if action_definition is None:
            self.logger.error("Unsupported action: Speech")
            return

            # Convert wav file to mp3 to save space
        episode_id = self.current_episode.episode_id
        dir_audio_out = os.path.join(self.dir_out, episode_id)

        if not os.path.exists(dir_audio_out):
            os.makedirs(dir_audio_out)

        time_start = time.time() - self.start_time
        file_wav = os.path.join(self.dir_out, file_name)
        file_mp3 = os.path.join(dir_audio_out, "%d.mp3" % time_start)
        sound = AudioSegment.from_wav(file_wav)
        sound.export(file_mp3, format="mp3")
        if os.path.isfile(file_wav):
            os.remove(file_wav)

        # Record action
        file_mp3 = os.path.join(os.path.basename(self.dir_out), episode_id, "%d.mp3" % time_start)
        action = Action_Audio(
            action_id=action_definition["action_id"],
            time_start=time_start,
            duration=1,
            utterance=utterance,
            file_name=file_mp3,
        )
        interaction = Interaction(agent_id=agent_id, action=action, is_object=False, status=None)
        self.add_interaction(interaction)
        self.to_broadcast["info"] = {"message": "%s: %s" % (agent_name, utterance)}

    def success(self):
        """
        Mark the end of an episode
        """
        if not self.is_ready:
            message = "Simulator was not initialized. Possible resolution: Start new episode."
            self.logger.error(message)
            raise Exception(message)
        value = self.__conditionally_add_stop(status=0)
        if value:
            self.logger.debug("Successful end of subtask in episode.")
            self.to_broadcast["info"] = {"message": "Success"}
        else:
            message = "No active episode or the active episode congtains no interactions."
            self.logger.warning(message)
            self.to_broadcast["info"] = {"message": message}
        return value

    def reset(self):
        """
        Reset the simulator to the initial state of self.current_episode
        """
        if not self.is_ready:
            message = "Simulator was not initialized. Possible resolution: Start new episode."
            self.logger.error(message)
            raise Exception(message)

        if self.current_episode is None:
            message = "Not in an active episode. Start a new one."
            self.logger.warning(message)
            raise Exception(message)

        self.go_to_pose(self.get_initial_pose(self.current_episode))
        self.current_episode.interactions = []
        self.to_broadcast["info"] = {"message": "Initial state"}

    def randomize_scene(self):
        """
        Randomize states and locations of pickupable objects in the current scene
        """
        if not self.is_ready:
            message = "Simulator was not initialized. Possible resolution: Start new episode."
            self.logger.error(message)
            raise Exception(message)

        if self.current_episode is None:
            message = "Not in an active episode. Start a new one."
            self.logger.warning(message)
            raise Exception(message)

        success1, msg1 = self.randomize_scene_objects_locations()
        success2, msg2 = self.randomize_scene_objects_states()
        self.current_episode.interactions = []
        self.to_broadcast["info"] = {"message": msg1 + msg2}

    def done(self, file_name=None):
        """
        Shut down the simulator and save the session with final simulator state; Should be called at end of collection/
        replay of episode
        :param file_name: If file_name is not None, the simulator session is saved in the same format as original games
        """
        if not self.is_ready:
            message = "Simulator was not initialized. Possible resolution: Start new episode."
            self.logger.error(message)
            raise Exception(message)

        if self.success():
            self.current_task.add_episode(copy.deepcopy(self.current_episode))
            self.save(file_name=file_name)  # Always save once episode is complete.
            self.logger.debug("End of episode.")
            self.to_broadcast["info"] = {"message": "Done"}
            self.shutdown_simulator()
        else:
            message = "No interactions in the current episode."
            self.logger.warning(message)
            self.to_broadcast["info"] = {"message": message}
        self.current_episode = None

    def set_target_object(self, oid, search):
        """
        Set the target object camera to face a particular sim object by id.
        :param oid: Valid object ID in the simulator or None
        :param search: If oid is None, search string to use to find an object by fuzzy matching
        """
        if not self.is_ready:
            message = "Simulator was not initialized. Possible resolution: Start new episode."
            self.logger.error(message)
            raise Exception(message)

        if self.current_episode is None:
            message = "Not in an active episode. Start a new one."
            self.logger.warning(message)
            raise Exception(message)

        obj_data = self.set_target_object_view(oid, search)
        if obj_data:
            return obj_data
        else:  # Failed to find such an object
            return {"success": False}

    def progress_check(self):
        """
        Check task progress in this episode. This is a wrapper for function check_episode_progress() that returns
        JSON-safe output.
        """
        if not self.is_ready:
            message = "Simulator was not initialized. Possible resolution: Start new episode."
            self.logger.error(message)
            raise Exception(message)

        if self.current_episode is None:
            message = "Not in an active episode. Start a new one."
            self.logger.warning(message)
            raise Exception(message)

        task_desc, success, subgoals = self.check_episode_progress(self.current_task)
        # Return JSON-safe encoding.
        for subgoal in subgoals:
            subgoal["success"] = 1 if subgoal["success"] else 0
            if "step_successes" in subgoal:
                subgoal["step_successes"] = [int(v) for v in subgoal["step_successes"]]
            for step in subgoal["steps"]:
                step["success"] = 1 if step["success"] else 0
        return {"task_desc": task_desc, "success": 1 if success else 0, "subgoals": subgoals}

    def preconditions_check(self):
        """
        Check task preconditions in this episode. This is a wrapper for check_episode_preconditions() that returns
        JSON-safe output
        """
        if not self.is_ready:
            message = "Simulator was not initialized. Possible resolution: Start new episode."
            self.logger.error(message)
            raise Exception(message)
        if self.current_episode is None:
            message = "Not in an active episode. Start a new one."
            self.logger.warning(message)
            raise Exception(message)

        met = self.check_episode_preconditions(self.current_task)
        # Return JSON-safe encoding.
        return {"met": 1 if met else 0}

    def record(self, record_mode=True):
        """
        Turn on record mode
        """
        if not self.is_ready:
            message = "Simulator was not initialized. Possible resolution: Start new episode."
            self.logger.error(message)
            raise Exception(message)

        try:
            if not self.is_record_mode:
                # Was previously not recording. Go to previous saved state.
                self.go_to_last_known_pose()

            if record_mode is None:
                record_mode = True
        except Exception as e:
            self.logger.error(str(e))
            raise e

        self.is_record_mode = record_mode
        self.logger.debug("Currently%s in record mode" % ("" if record_mode else " not"))

    def info(self, include_scenes=False, include_objects=False):
        """
        Obtain information about the current task and episode
        """
        num_tasks = 0 if self.current_task is None else len(self._dataset.tasks)
        comments_task = ""
        if num_tasks > 0:
            task_id = self._dataset.tasks[-1].task_id
            comments_task = self._dataset.tasks[-1].comments
        else:
            task_id = -1

        num_episodes = 0
        episode_id = ""
        history_utterance = []
        if self.current_task is not None:
            num_episodes = len(self.current_task.episodes)
            if self.current_episode is not None:
                episode_id = self.current_episode.episode_id
                for interaction in self.current_episode.interactions:
                    action_name = self._dataset.definitions.map_actions_id2info[interaction.action.action_id][
                        "action_name"
                    ]
                    if action_name == "Text":
                        history_utterance.append(
                            {
                                "agent_id": interaction.agent_id,
                                "utterance": interaction.action.utterance,
                                "action_type": interaction.action.action_type,
                            }
                        )
                    elif action_name == "Speech":
                        history_utterance.append(
                            {
                                "agent_id": interaction.agent_id,
                                "utterance": interaction.action.utterance,
                                "action_type": interaction.action.action_type,
                            }
                        )
            if num_episodes > 0:
                episode_id = self.current_task.episodes[-1].episode_id

        basic_info = {
            "comments_task": comments_task,
            "task_id": task_id,
            "num_tasks": num_tasks,
            "num_episodes": num_episodes,
            "episode_id": episode_id,
            "history_utterance": history_utterance,
            "is_record_mode": 1 if self.is_record_mode else 0,
        }

        if include_scenes:
            basic_info.update(self.get_available_scenes())

        if include_objects:
            basic_info.update(self.get_available_objects())

        return basic_info

    def save(self, file_name=None):
        """
        Save episode to file
        :param file_name: File name to save episode to; if None, a random file name is assigned
        """
        logger.info("simulator_base save called with file_name " + str(file_name))
        if file_name is None or len(file_name) < 1:
            file_name = self.prefix + "_" + ".json"

        if not os.path.exists(self.dir_out):
            os.makedirs(self.dir_out)

        filepath = os.path.join(self.dir_out, file_name)
        data = self._dataset.to_dict()
        save_dict_as_json(data, filepath)

        self.logger.debug("Saved: %s" % filepath)

    def save_scene_state(self):
        """
        Save simulator state: floor plan, object locations and states
        """
        s = self.get_scene_object_locs_and_states()
        if not os.path.exists(self.dir_out):
            os.makedirs(self.dir_out)
        file_name = os.path.join(self.dir_out, self.prefix + "_state" + ".json")
        with open(file_name, "w") as f:
            json.dump(s, f)
        self.logger.debug("Saved: %s" % file_name)
        return file_name

    def load_scene_state(self, fn=None, init_state=None):
        """
        Reset start time and init state.
        :param fn: Filename to load initial state from
        :param init_state: Valid initial state to initialize simulator with; must be an instance of class
        Initialization in dataset.py
        """
        assert fn is None or init_state is None
        file_name = None
        if fn is not None or (fn is None and init_state is None):
            # If no init conditions to read set, try to guess the last state file written out.
            file_name = os.path.join(self.dir_out, self.prefix + "_state" + ".json") if fn is None else fn
            if os.path.isfile(file_name):
                with open(file_name, "r") as f:
                    init_state = json.load(f)
            else:
                raise FileNotFoundError('No scene save file "%s" found' % file_name)

        self.start_time = time.time()
        success1, msg1 = self.restore_scene_object_locs_and_states(
            init_state.objects if type(init_state) == Initialization else init_state["objects"]
        )
        success2, msg2 = self.set_agent_poses(
            init_state.agents if type(init_state) == Initialization else init_state["agents"]
        )
        self.set_init_state()
        self.start_time = time.time()
        if success1 and success2:
            self.logger.debug(
                "Loaded from %s" % file_name if file_name is not None else "Loaded from supplied init state arg"
            )
        else:
            self.logger.debug(
                "Error when loading: %s" % file_name if file_name is not None else "Error loading init state"
            )
            if not success1:
                self.logger.debug("restore_scene_object_locs_and_states failed with message " + msg1)
            if not success2:
                self.logger.debug("set_agent_poses failed with message " + msg2)
        return file_name, success1 and success2

    def get_json(self):
        """
        Return current task and episode information in a JSON format
        """
        return self._dataset.to_dict()

    def get_available_scenes(self):
        raise NotImplementedError("Derived class must implement this!")

    def get_available_objects(self):
        raise NotImplementedError("Derived class must implement this!")

    def get_hotspots(self, agent_id):
        raise NotImplementedError("Derived class must implement this!")

    def check_episode_progress(self, task):
        raise NotImplementedError("Derived class must implement this!")

    def set_target_object_view(self, oid, search):
        raise NotImplementedError("Derived class must implement this!")

    def get_target_object_seg_mask(self, oid):
        raise NotImplementedError("Derived class must implement this!")

    def check_episode_preconditions(self, task):
        raise NotImplementedError("Derived class must implement this!")

    def shutdown_simulator(self):
        raise NotImplementedError("Derived class must implement this!")

    def teleport_agent_to_face_object(self, obj, agent_id, force_face=None):
        raise NotImplementedError("Derived class must implement this!")

    def get_initial_pose(self, episode):
        """
        Return initial positions of agents in an episode
        :param episode: instance of class Episode in dataset.py
        """
        if episode is None:
            return None

        if episode.initial_state is None:
            return None

        if len(episode.initial_state.agents) < 1:
            return None

        for agent in episode.initial_state.agents:
            return agent.pose

        return None

    def get_current_pose(self):
        raise NotImplementedError("Derived class must implement this!")

    def go_to_initial_state(self, episode):
        """
        Move agents to their positions at start of episode
        :param episode: instance of class Episode defined in dataset.py
        """
        pose = self.get_initial_pose(episode)
        if pose is None:
            return False

        self.go_to_pose(pose)
        return True

    def go_to_last_known_pose(self):
        """
        Return agents to the most recent pose recorded in a past interaction
        """
        last_known_pose = None
        if self.current_episode is not None:
            for interaction in self.current_episode.interactions[::-1]:
                if hasattr(interaction.action, "pose"):
                    # Found it ion a previous "Motion" interaction
                    last_known_pose = interaction.action.pose
                    break
            if last_known_pose is None:
                # Could not find in any of the interactions in the current episode.
                # Go to the initial state.
                last_known_pose = self.get_initial_pose(self.current_episode)

        self.go_to_pose(last_known_pose)

    def go_to_pose(self, pose):
        return

    def randomize_scene_objects_locations(self):
        raise NotImplementedError("Derived class must implement this!")

    def randomize_scene_objects_states(self):
        raise NotImplementedError("Derived class must implement this!")

    def randomize_agent_positions(self):
        raise NotImplementedError("Derived class must implement this!")

    def restore_initial_state(self):
        raise NotImplementedError("Derived class must implement this!")

    def get_scene_object_locs_and_states(self):
        raise NotImplementedError("Derived class must implement this!")

    def restore_scene_object_locs_and_states(self, objects):
        raise NotImplementedError("Derived class must implement this!")

    def set_agent_poses(self, agents):
        raise NotImplementedError("Derived class must implement this!")

    def set_init_state(self):
        self.start_time = time.time()

    def get_current_state(self):
        raise NotImplementedError("Derived class must implement this!")

    def get_latest_images(self):
        raise NotImplementedError("Derived class must implement this!")

    def export_images(self, force=False):
        """
        Return encoded image frames for data collection
        """
        latest_images = self.get_latest_images()
        imgs = {}
        for name in latest_images:
            if name in self.live_feeds:
                # If this is a feed we're subscribed to and the frame values have changed, encode.
                if force or name not in self.last_images or not np.all(self.last_images[name] == latest_images[name]):
                    enc_img_str = self.encode_image(latest_images[name])
                    imgs[name] = enc_img_str

        self.last_images = latest_images
        return imgs

    def encode_image(self, img_as_np_array):
        """
        Convert image to bytes
        :param img_as_np_array: Image to be encoded, of type numpy.ndarray
        """
        pil_img = Image.fromarray(img_as_np_array)
        buff = io.BytesIO()
        pil_img.save(buff, format="jpeg")
        enc_img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        return enc_img_str

    def __conditionally_add_stop(self, status=None):
        """
        Add a Stop action to the end of the current episode
        """
        if (self.current_task is None) or (self.current_episode is None):
            return False

        if len(self.current_episode.interactions) < 1:
            return False

        last_interaction = self.current_episode.interactions[-1]

        if last_interaction.agent_id == 0:
            # Last action was taken by the user.
            if status is not None:
                last_interaction.status = status
            return True

        if last_interaction.action.action_type != "Motion":
            # Last action type was not Motion.
            if status is not None:
                last_interaction.status = status
            return True

        action_id_stop = 0
        if last_interaction.action.action_id == action_id_stop:
            # Last action was already "Stop".
            if status is not None:
                last_interaction.status = status
            return True

        action = Action_Motion(
            action_id=action_id_stop,
            time_start=time.time() - self.start_time,
            duration=1,
            pose=Pose(0, 0, 0, 0, 0, 0),
            pose_delta=Pose(0, 0, 0, 0, 0, 0),
        )
        interaction = Interaction(agent_id=1, action=action, is_object=False, status=status)
        self.add_interaction(interaction)

        return True

    def __generate_random_image_bytes(self, dim=(540, 960, 3)):
        img = Image.fromarray(np.zeros(dim, dtype=np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="jpeg")
        return img_bytes.getvalue()

    def __reset_helper(self, dir_out=None):
        self.is_record_mode = True
        self.current_episode = None
        self.current_task = None
        self.is_ready = False
        self.dir_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out") if dir_out is None else dir_out
        if not os.path.exists(self.dir_out):
            os.makedirs(self.dir_out)

        self.prefix = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        self.start_time = time.time()

        img = Image.fromarray(np.random.randint(0, 255, size=(512, 910, 3), dtype=np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="jpeg")
        self.to_broadcast = {
            "ego": self.__generate_random_image_bytes(),
            "allo": self.__generate_random_image_bytes(),
            "map": self.__generate_random_image_bytes(),
            "semantic": self.__generate_random_image_bytes(),
            "info": {"message": ""},
        }
