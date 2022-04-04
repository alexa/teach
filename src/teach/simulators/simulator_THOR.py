# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import copy
import importlib.resources
import json
import logging
import os
import platform
import random
import tempfile
import time

import cv2
import networkx as nx
import numpy as np
from ai2thor.build import arch_platform_map, build_name
from ai2thor.controller import Controller
from fuzzywuzzy import fuzz

import teach.meta_data_files.ai2thor_resources as ai2thor_resources
import teach.meta_data_files.config as config_directory
from teach.dataset.initialization import Initialization
from teach.dataset.pose import Pose
from teach.logger import create_logger
from teach.settings import get_settings
from teach.simulators.simulator_base import SimulatorBase

# Commit where FillLiquid bug is fixed: https://github.com/allenai/ai2thor/issues/844
COMMIT_ID = "fdc047690ee0ab7a91ede50d286bd387d379713a"

# debug manual flag
debug_print_all_sim_steps = False

logger = create_logger(__name__)


class TEAChController(Controller):
    def __init__(self, base_dir: str, **kwargs):
        self._base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        super().__init__(**kwargs)

    @staticmethod
    def build_local_executable_path(base_dir: str, commit_id: str, release_dir: str = "releases"):
        """Helper method to build the path to the local executable. Useful when executable is pre-downloaded."""
        arch = arch_platform_map[platform.system()]
        name = build_name(arch, commit_id)
        return os.path.join(base_dir, release_dir, name, name)

    @staticmethod
    def base_dir_in_tmp():
        tempdir = tempfile.gettempdir()
        base_dir = os.path.join(tempdir, "ai2thor")
        os.makedirs(base_dir, exist_ok=True)

        return base_dir

    @property
    def base_dir(self):
        return self._base_dir


class SimulatorTHOR(SimulatorBase):
    def __init__(
        self,
        task_type="eqa_complex",
        comments=None,
        fps=25,
        logger_name=__name__,
        logger_level=logging.DEBUG,
        dir_out=None,
        s3_bucket_name=None,
        web_window_size=900,
        commander_embodied=False,
        visibility_distance=1.5,
    ):
        """
        Constructor for Simulator_THOR - a wrapper over AI2-THOR

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

        :param web_window_size: Window/ image sizes (square) to be used by simulator; 900 for TEACh data collection
        :type web_window_size: Int

        :param commander_embodied: True if the Commander should also be allowed to interact with objects; False for
        TEACh data collection
        :type commander_embodied: Bool

        :param visibility_distance: Max distance an agent can be from an object to successfully interact with it; 1.5
        for TEACh data collection
        :type visibility_distance: Float
        """
        time_start = time.time()
        super().__init__(
            task_type,
            comments,
            fps=fps,
            logger_name=logger_name,
            logger_level=logger_level,
            dir_out=dir_out,
            s3_bucket_name=s3_bucket_name,
        )
        time_base_init = time.time()
        logger.info("Initializing simulator... time to init Simulator_base: %s sec" % (time_base_init - time_start))
        self.controller = None

        teach_settings = get_settings()
        self.controller_base_dir = teach_settings.AI2THOR_BASE_DIR
        use_local_exe = teach_settings.AI2THOR_USE_LOCAL_EXE
        self.controller_local_executable_path = (
            TEAChController.build_local_executable_path(self.controller_base_dir, COMMIT_ID) if use_local_exe else None
        )

        self.world_type = "Kitchen"
        self.world = None
        self.grid_size = 0.25
        self.hotspot_pixel_width = 10
        self.web_window_size = web_window_size
        self.commander_embodied = commander_embodied
        self.randomize_object_search = False
        self.visibility_distance = visibility_distance
        self.object_target_camera_idx = None
        self.navigation_graph = self.navigation_points = None
        self.topdown_cam_orth_size = self.topdown_lower_left_xz = None  # Used for MapGoals
        self.floor_oid = None  # used for handoffs to temporarily store objects on the floor

        # The following is a dictionary for custom object metadata. When adding custom object properties, DO NOT use
        # property names already used by AI2-THOR. If the same property is needed here, prefix the property name with
        # the project for which you are using it. For example, the AI2-THOR property isSliced could be changed to
        # simbotIsSliced if the project simbot needed custom behaviour from isSliced
        self.__custom_object_metadata = dict()

        # Affordances by action type - identifies what properties an object must satisfy for it to be possible to take
        # an action on it; Used in highlighting valid objects in TEACh data collection interface to assist annotators
        self.action_to_affordances = {
            "Pickup": [{"pickupable": True, "isPickedUp": False}],
            "Place": [{"receptacle": True}],
            "Open": [{"openable": True, "isOpen": False}],
            "Close": [{"openable": True, "isOpen": True}],
            "ToggleOn": [{"toggleable": True, "isToggled": False}],
            "ToggleOff": [{"toggleable": True, "isToggled": True}],
            "Slice": [{"sliceable": True, "isSliced": False}],
            "Dirty": [{"dirtyable": True, "isDirty": False}],
            "Clean": [{"dirtyable": True, "isDirty": True}],
            "Fill": [{"canFillWithLiquid": True, "isFilledWithLiquid": False}],
            "Empty": [{"canFillWithLiquid": True, "isFilledWithLiquid": True}],
            "Pour": [
                {"canFillWithLiquid": True, "isFilledWithLiquid": False},
                {"objectType": "Sink"},
                {"objectType": "SinkBasin"},
                {"objectType": "Bathtub"},
                {"objectType": "BathtubBasin"},
            ],
            "Break": [{"breakable": True, "isBroken": False}],
        }
        time_end = time.time()
        logger.info("Finished initializing simulator. Total time: %s sec" % (time_end - time_start))

    def set_task(self, task, task_params=None, comments=""):
        """
        Set the current task to provided Task_THOR object
        Tasks are defined in json files under task_definitions
        :param task: instance of Task_THOR class
        :param task_params list of parameters to the task, possibly empty; must match definition nparams in length
        :param comments: Informative comments for the current task. Default = ''
        :type comments: String
        """
        logger.debug("Setting task = %s" % str(task))
        new_task = copy.deepcopy(task)
        if task_params is not None:
            new_task.task_params = task_params
        new_task.comments = comments
        new_task.episodes = [] if self.current_episode is None else [self.current_episode]
        self._dataset.add_task(new_task)
        self.current_task = new_task
        self.logger.debug("New task: %d, %s, %s, %s" % (task.task_id, task.task_name, comments, str(task.task_params)))
        self.to_broadcast["info"] = {"message": ""}
        logger.info("SimulatorTHOR set_task done New task: %d, %s, %s" % (task.task_id, task.task_name, comments))

    def set_task_by_id(self, task_id: int, task_params=None, comments=""):
        """
        Set the current task to task defined in default_definitions.json with provided task_id
        :param task_id: task id number from task definition json file
        :param task_params list of parameters to the task, possibly empty; must match definition nparams in length
        :param comments: Informative comments for the current task. Default = ''
        :type comments: String
        """
        task = self._dataset.definitions.map_tasks_id2info[task_id]
        task.task_params = task_params
        self.set_task(task=task, task_params=task_params, comments=comments)

    def set_task_by_name(self, task_name: str, task_params=None, comments=""):
        """
        Set the current task to task defined in default_definitions.json with provided task_name
        :param task_name task name from task definition json file
        :param task_params list of parameters to the task, possibly empty; must match definition nparams in length
        :param comments: Informative comments for the current task. Default = ''
        :type comments: String
        """
        task = self._dataset.definitions.map_tasks_name2info[task_name]
        task.task_params = task_params
        self.set_task(task=task, task_params=task_params, comments=comments)

    def __add_obj_classes_for_objs(self):
        """
        For each object in AI2-THOR metadata, update with manually defined object classes to be tracked in custom
        properties
        """
        # Load custom object classes
        with importlib.resources.open_text(ai2thor_resources, "custom_object_classes.json") as file_handle:
            custom_object_classes = json.load(file_handle)
        # Assign custom classes to each object
        all_objects = self.get_objects(self.controller.last_event)
        for obj in all_objects:
            cur_obj_classes = [obj["objectType"]]
            if obj["objectType"] == "Sink":
                cur_obj_classes += ["SinkBasin"]
            if obj["objectType"] == "SinkBasin":
                cur_obj_classes += ["Sink"]
            if obj["objectType"] == "Bathtub":
                cur_obj_classes += ["BathtubBasin"]
            if obj["objectType"] == "BathtubBasin":
                cur_obj_classes += ["Bathtub"]
            if obj["objectType"] in custom_object_classes:
                cur_obj_classes += custom_object_classes[obj["objectType"]]
            self.__update_custom_object_metadata(obj["objectId"], "simbotObjectClass", cur_obj_classes)

    def __init_custom_object_metadata(self):
        """
        Reset custom object metadata to initial state: erase previously tracked properties, add manual classes for all
        objects and check for custom property updates from current state
        """
        self.__custom_object_metadata = dict()
        self.__add_obj_classes_for_objs()
        self.__check_per_step_custom_properties()

    def __check_per_step_custom_properties(self, objs_before_step=None):
        """
        Check whether any custom object properties need to be updated; Should be called after taking each action
        """
        # Update whether things got cleaned and filled with water
        self.__update_sink_interaction_outcomes(self.controller.last_event)
        # Update whether a mug should be filled with coffee
        self.__update_custom_coffee_prop(self.controller.last_event, objs_before_step)
        # Update whether things got cooked
        self.__update_custom_property_cooked(self.controller.last_event)
        # Check for objects that are boiled at the start of the episode
        self.__update_custom_property_boiled(objs_before_step, self.controller.last_event)

    def __update_custom_object_metadata(self, object_id, custom_property_name, custom_property_value):
        """
        Update custom properties
        """
        if object_id not in self.__custom_object_metadata:
            self.__custom_object_metadata[object_id] = dict()
        self.__custom_object_metadata[object_id][custom_property_name] = custom_property_value

    def __append_to_custom_object_metadata_list(self, object_id, custom_property_name, custom_property_value):
        """
        Add values to custom properties that are lists
        """
        if object_id not in self.__custom_object_metadata:
            self.__custom_object_metadata[object_id] = dict()
        if custom_property_name not in self.__custom_object_metadata[object_id]:
            self.__custom_object_metadata[object_id][custom_property_name] = list()
        if custom_property_value not in self.__custom_object_metadata[object_id][custom_property_name]:
            self.__custom_object_metadata[object_id][custom_property_name].append(custom_property_value)

    def __delete_from_custom_object_metadata_list(self, object_id, custom_property_name, custom_property_value):
        """
        Delete values from custom properties that are lists
        """
        if (
            object_id in self.__custom_object_metadata
            and custom_property_name in self.__custom_object_metadata[object_id]
            and custom_property_value in self.__custom_object_metadata[object_id][custom_property_name]
        ):
            del self.__custom_object_metadata[object_id][custom_property_name][
                self.__custom_object_metadata[object_id][custom_property_name].index(custom_property_value)
            ]

    def __delete_object_from_custom_object_metadata(self, object_id):
        """
        Delete custom properties of an object
        :param object_id: ID of object whose properties are to be deleted
        """
        if object_id in self.__custom_object_metadata:
            del self.__custom_object_metadata[object_id]
        for oid in self.__custom_object_metadata:
            for prop in self.__custom_object_metadata[oid]:
                if (
                    type(self.__custom_object_metadata[oid][prop]) is list
                    and object_id in self.__custom_object_metadata[oid][prop]
                ):
                    del self.__custom_object_metadata[oid][prop][
                        self.__custom_object_metadata[oid][prop].index(object_id)
                    ]
                elif object_id == self.__custom_object_metadata[oid][prop]:
                    self.__custom_object_metadata[oid][prop] = None

    def __transfer_custom_metadata_on_slicing_cracking(self, objects):
        """
        When objects get sliced or cracked, their object IDs change because one object may become multiple objects.
        Transfer custom properties from the original object to the new object(s)
        :param objects: Output of get_objects()
        """
        objects_to_delete = set()
        for obj in objects:
            transfer_needed = False
            orig_obj_id = None
            if "Sliced" in obj["objectId"]:
                transfer_needed = True
                orig_obj_id = "|".join(obj["objectId"].split("|")[:-1])
            if "Cracked" in obj["objectId"]:
                transfer_needed = True
                orig_obj_id = "|".join(obj["objectId"].split("|")[:-1])

            if transfer_needed and orig_obj_id is not None and orig_obj_id in self.__custom_object_metadata:
                self.__custom_object_metadata[obj["objectId"]] = copy.deepcopy(
                    self.__custom_object_metadata[orig_obj_id]
                )
                if (
                    "simbotLastParentReceptacle" in self.__custom_object_metadata[obj["objectId"]]
                    and self.__custom_object_metadata[obj["objectId"]]["simbotLastParentReceptacle"] is not None
                ):
                    poid = self.__custom_object_metadata[obj["objectId"]]["simbotLastParentReceptacle"]
                    self.__append_to_custom_object_metadata_list(poid, "simbotIsReceptacleOf", obj["objectId"])
                objects_to_delete.add(orig_obj_id)
        for obj_id in objects_to_delete:
            self.__delete_object_from_custom_object_metadata(obj_id)

    def get_objects(self, event=None):
        """
        Return objects augmented by custom properties
        :param event: Simulator event to be used to obtain object properties, usually self.controller.last_event to get
        current object states
        """
        if event is None:
            if self.commander_embodied:
                event = self.controller.last_event.events[0]
            else:
                event = self.controller.last_event

        for obj in event.metadata["objects"]:
            if obj["objectId"] in self.__custom_object_metadata:
                obj.update(self.__custom_object_metadata[obj["objectId"]])
        return event.metadata["objects"]

    def get_inventory_objects(self, event):
        """
        Return objects held in hand by agents
        :param event: Simulator event to be used to obtain object properties, usually self.controller.last_event to get
        current object states
        """
        for obj in event.metadata["inventoryObjects"]:
            if obj["objectId"] in self.__custom_object_metadata:
                obj.update(self.__custom_object_metadata[obj["objectId"]])
        return event.metadata["inventoryObjects"]

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
        Start a new episode in a random scene
        :param world: AI2-THOR floor plan to be used or None; if None a random scene (matching specified world_type
        if provided) is used
        :param world_type: One of "Kitchen", "Bedroom", "Bathroom", "Living room" or None; if world is None and
        world_type is specified, a random world of the specified world_type is used
        :param object_tuples: Used to specify initial states of objects
        :param commander_embodied: True if the Commander should also be allowed to interact with objects; False for
        TEACh data collection
        :param episode_id: Used to specify a custom episode ID
        :param randomize_object_search: If True, attempts to search for objects will return a random object of type
        matching the search string; if false, the object closest to the agent is always returned on search
        """
        logger.info("In simulator_THOR.start_new_episode, world = %s world_type = %s" % (world, world_type))
        self.randomize_object_search = randomize_object_search
        if commander_embodied is not None:
            self.commander_embodied = commander_embodied
        else:
            self.commander_embodied = False
            logger.info("SimulatorTHOR warning: commander_embodied was not set on first episode init; default to False")
        if world is None:
            world_type, world = self.select_random_world(world_type=world_type)

        super().start_new_episode(
            world=world,
            world_type=world_type,
            object_tuples=object_tuples,
            commander_embodied=commander_embodied,
            episode_id=episode_id,
            randomize_object_search=randomize_object_search,
        )

        logger.info("In SimulatorTHOR.start_new_episode, before __launch_simulator")
        self.__launch_simulator(world=world, world_type=world_type)
        logger.info("In SimulatorTHOR.start_new_episode, completed __launch_simulator")
        self.__init_custom_object_metadata()
        state = self.get_scene_object_locs_and_states()
        self.current_episode.initial_state = Initialization(
            time_start=0,
            agents=state["agents"],
            objects=state["objects"],
            custom_object_metadata=self.__custom_object_metadata,
        )

    def save(self, file_name=None):
        """
        Save the session using the current state as the final simulator state. This does not shut down the simulator.
        Call done() instead if simulator should be shut down after this
        :param file_name: If file_name is not None, the simulator session is saved in the same format as original games
        """
        # Add final state to log.
        state = self.get_scene_object_locs_and_states()
        self.current_episode.final_state = Initialization(
            time_start=time.time() - self.start_time,
            agents=state["agents"],
            objects=state["objects"],
            custom_object_metadata=self.__custom_object_metadata,
        )

        # Save log file
        super().save(file_name=file_name)

    def done(self, file_name=None):
        """
        Shut down the simulator and save the session with final simulator state; Should be called at end of collection/
        replay of an episode
        :param file_name: If file_name is not None, the simulator session is saved in the same format as original games
        """
        # Add final state to log.
        state = self.get_scene_object_locs_and_states()
        self.current_episode.final_state = Initialization(
            time_start=time.time() - self.start_time,
            agents=state["agents"],
            objects=state["objects"],
            custom_object_metadata=self.__custom_object_metadata,
        )

        # End AI2-THOR Unity process
        self.controller.stop()
        self.controller = None

        # Save log file and change current_episode metadata in the base
        super().done(file_name=file_name)

    def __argmin(self, lst):
        """
        Return the index of the least element in l
        """
        return lst.index(min(lst))

    def __get_nearest_object_face_to_position(self, obj, pos):
        """
        Examine the AI2-THOR property 'axisAlignedBoundingBox'['cornerPoints'] and return the pose closest to target
        pose specified in param pos
        :param obj: the object to examine the faces of
        :param pos: the target position to get near
        """
        coords = ["x", "y", "z"]
        if obj["pickupable"]:
            # For pickupable objects we don't actually need to examine corner points and doing so sometimes causes
            # errors with clones
            return obj["position"]
        xzy_obj_face = {
            c: obj["axisAlignedBoundingBox"]["cornerPoints"][
                self.__argmin(
                    [
                        np.abs(obj["axisAlignedBoundingBox"]["cornerPoints"][pidx][coords.index(c)] - pos[c])
                        for pidx in range(len(obj["axisAlignedBoundingBox"]["cornerPoints"]))
                    ]
                )
            ][coords.index(c)]
            for c in coords
        }
        return xzy_obj_face

    def __aim_camera_at_object(self, obj, camera_id):
        """
        Position camera specified by camera_id such that object obj is visible; Used to set target object view for
        TEACh data collection interface
        :param obj: Object to face - an element of the output of get_objects()
        :param camera_id: A valid camera ID
        """
        nav_point_idx = self.__get_nav_graph_point(obj["position"]["x"], obj["position"]["z"])
        face_obj_rot = self.__get_nav_graph_rot(
            self.navigation_points[nav_point_idx]["x"],
            self.navigation_points[nav_point_idx]["z"],
            obj["position"]["x"],
            obj["position"]["z"],
        )

        # Calculate the angle at which to look at the object to center it.
        # We look from the head height of the agent [https://github.com/allenai/ai2thor/issues/266]
        # Head gaze is the hypotenuse of a right triangle whose legs are the xz (floor) distance to the obj and the
        # difference in gaze versus object height.
        # To get the object 'face' instead of center (which could be out of frame, especially for large objects like
        # drawers and cabinets), we decide the x,z,y position of the obj as the min distance to its corners.
        xzy_obj_face = self.__get_nearest_object_face_to_position(obj, self.navigation_points[nav_point_idx])
        xz_dist = np.sqrt(
            np.power(xzy_obj_face["x"] - self.navigation_points[nav_point_idx]["x"], 2)
            + np.power(xzy_obj_face["z"] - self.navigation_points[nav_point_idx]["z"], 2)
        )
        y_diff = 1.8 - xzy_obj_face["y"]
        theta = np.arctan(y_diff / xz_dist) * 180.0 / np.pi if not np.isclose(xz_dist, 0) else 0

        action = dict(
            action="UpdateThirdPartyCamera",
            thirdPartyCameraId=camera_id,
            rotation=dict(x=theta, y=self.__get_y_rot_from_xz(face_obj_rot[0], face_obj_rot[1]), z=0),
            position=dict(
                x=self.navigation_points[nav_point_idx]["x"], y=1.8, z=self.navigation_points[nav_point_idx]["z"]
            ),
        )
        if debug_print_all_sim_steps:
            logger.info("step %s", action)
        self.controller.step(action)
        return nav_point_idx, face_obj_rot

    def teleport_agent_to_face_object(self, obj, agent_id, force_face=None, get_closest=True):
        """
        Move agent to a position where object obj is visible
        :param obj: Object to face - an element of the output of get_objects()
        :param agent_id: 0 for Commander and 1 for Driver/ Follower
        :param force_face: Specify a particular target rotation
        :param get_closest: If True the agent is always places at closest position; if false, nucleus sampling within
        a distance radius around the target object is used
        """
        # Get point and facing direction.
        tried_points = set()
        face_obj_rot = nav_point_idx = None
        while face_obj_rot is None or (force_face is not None and face_obj_rot != force_face):
            nav_point_idx = self.__get_nav_graph_point(
                obj["position"]["x"], obj["position"]["z"], exclude_points=tried_points, get_closest=get_closest
            )
            if nav_point_idx is None:
                return False, None, None
            face_obj_rot = self.__get_nav_graph_rot(
                self.navigation_points[nav_point_idx]["x"],
                self.navigation_points[nav_point_idx]["z"],
                obj["position"]["x"],
                obj["position"]["z"],
            )
            tried_points.add(nav_point_idx)
        if force_face is not None and force_face != face_obj_rot:
            return False, nav_point_idx, face_obj_rot

        # Teleport
        agent_pose = (
            self.controller.last_event.events[agent_id].metadata["agent"]
            if self.commander_embodied
            else self.controller.last_event.metadata["agent"]
        )
        action = dict(
            action="Teleport",
            agentId=agent_id,
            rotation=dict(
                x=agent_pose["rotation"]["x"],
                y=self.__get_y_rot_from_xz(face_obj_rot[0], face_obj_rot[1]),
                z=agent_pose["rotation"]["z"],
            ),
            position=dict(
                x=self.navigation_points[nav_point_idx]["x"],
                y=agent_pose["position"]["y"],
                z=self.navigation_points[nav_point_idx]["z"],
            ),
            horizon=0,
        )
        if debug_print_all_sim_steps:
            logger.info("step %s", action)
        event = self.controller.step(action)
        if not event.metadata["lastActionSuccess"]:
            return False, nav_point_idx, face_obj_rot
        return True, nav_point_idx, face_obj_rot

    def obj_dist_to_nearest_agent(self, obj):
        """
        Return Euclidean distance between a given object and the nearest agent in the sim.
        """
        if self.commander_embodied:
            # For immobile commander, only check what object is closest to driver.
            events = [self.controller.last_event.events[0]]
        else:
            events = [self.controller.last_event]
        ds = [
            np.linalg.norm(
                [
                    obj["position"]["x"] - e.metadata["agent"]["position"]["x"],
                    obj["position"]["y"] - e.metadata["agent"]["position"]["y"],
                    obj["position"]["z"] - e.metadata["agent"]["position"]["z"],
                ]
            )
            for e in events
        ]
        return min(ds)

    def __agent_dist_to_agent(self, agent_id_a, agent_id_b):
        """
        Return Euclidean distance between two agents in the sim.
        """
        a_agent_pos = self.controller.last_event.events[agent_id_a].metadata["agent"]["position"]
        b_agent_pos = self.controller.last_event.events[agent_id_b].metadata["agent"]["position"]
        return np.linalg.norm([a_agent_pos[c] - b_agent_pos[c] for c in ["x", "y", "z"]])

    def check_episode_preconditions(self, task):
        """
        Check whether the current simulator state is one in which the input task can be completed
        :param task: Instance of Task_THOR; task to be checked
        """
        return task.check_episode_preconditions(self, self.get_objects(self.controller.last_event))

    def check_episode_progress(self, task):
        """
        Check completion status of input task given the current simulator state
        :param task: Instance of Task_THOR; task to be checked
        :return: (task_desc:str, success:bool, subgoal_status:list)
                 Each element of subgoal_status is a dict with keys 'success':bool, 'description':str and 'steps':list
                 Each element of subgoal_status[idx]['steps'] is a dict with keys 'success':bool, 'objectId':str,
                     'objectType':str, 'desc':str
        """
        progress_check_output = task.check_episode_progress(self.get_objects(self.controller.last_event), self)
        return (
            progress_check_output["description"],
            progress_check_output["success"],
            progress_check_output["subgoals"],
            progress_check_output["goal_conditions_total"],
            progress_check_output["goal_conditions_satisfied"],
        )

    def __get_nearest_object_matching_search_str(self, query, exclude_inventory=False):
        """
        Obtain the nearest object to the commander OR driver matching the given search string.
        :param query: the search string to check against AI2-THOR objectType of objects (uses fuzzy matching)
        :param exclude_inventory: if True, don't include inventory objects as candidates (e.g., nothing held will return)
        """
        closest_obj = closest_str_ratio = closet_obj_d_to_agent = None
        if self.commander_embodied:
            le = self.controller.last_event.events[0]
            inv_objs = self.get_inventory_objects(self.controller.last_event.events[0])
            inv_objs.extend(self.get_inventory_objects(self.controller.last_event.events[1]))
        else:
            le = self.controller.last_event
            inv_objs = self.get_inventory_objects(le)
        inv_obj_ids = [o["objectId"] for o in inv_objs]
        for obj in le.metadata["objects"]:
            if exclude_inventory and obj["objectId"] in inv_obj_ids:
                logger.info("%s in inv; skipping" % obj["objectId"])
                continue
            str_ratio = fuzz.ratio(obj["objectType"], query)
            if (
                str_ratio > 0
                and
                # Closer string match or equal string match but closer to agent
                (
                    closest_obj is None
                    or str_ratio > closest_str_ratio
                    or
                    # Physically closer to closest agent.
                    (str_ratio == closest_str_ratio and self.obj_dist_to_nearest_agent(obj) < closet_obj_d_to_agent)
                )
            ):
                closest_obj = obj
                closest_str_ratio = str_ratio
                closet_obj_d_to_agent = self.obj_dist_to_nearest_agent(obj)
        return closest_obj

    def __get_random_object_matching_search_str(self, query, exclude_inventory=False):
        """
        Obtain a random object to the commander OR driver matching the given search string.
        :param query: the search string to check against AI2-THOR objectType of objects (uses fuzzy matching)
        :param exclude_inventory: if True, don't include inventory objects as candidates (e.g., nothing held will return)
        """
        if self.commander_embodied:
            le = self.controller.last_event.events[0]
            inv_objs = self.get_inventory_objects(self.controller.last_event.events[0])
            inv_objs.extend(self.get_inventory_objects(self.controller.last_event.events[1]))
        else:
            le = self.controller.last_event
            inv_objs = self.get_inventory_objects(le)
        inv_obj_ids = [o["objectId"] for o in inv_objs]
        candidate_objects = self.get_objects(le)
        if exclude_inventory:
            candidate_objects = [obj for obj in candidate_objects if obj["objectId"] not in inv_obj_ids]

        str_ratios = [fuzz.ratio(obj["objectType"], query) for obj in candidate_objects]
        max_ratio = np.max(str_ratios)
        max_ratio_idxs = [idx for idx in range(len(str_ratios)) if np.isclose(max_ratio, str_ratios[idx])]
        closest_match_objects = [candidate_objects[idx] for idx in max_ratio_idxs]
        return np.random.choice(closest_match_objects)

    def get_target_object_seg_mask(self, oid):
        """
        Get a numpy array with 1s on oid segmentation mask and 0s elsewhere.
        :param oid: ID of object to be highlighted in the mask
        """
        r = self.get_hotspots(
            agent_id=None, camera_id=self.object_target_camera_idx, object_id=oid, return_full_seg_mask=True
        )
        return r

    def set_target_object_view(self, oid, search):
        """
        Move target object third party camera to look at specified objectId and returns associated hotspots
        :param oid: ID of object to be shown or None
        :param search: if oid is None, search string to use for fuzzy matching of object type
        """
        assert oid is None or search is None
        le = self.controller.last_event.events[0] if self.commander_embodied else self.controller.last_event
        if oid is None:  # need to choose an oid via search first
            if self.randomize_object_search:
                obj = self.__get_random_object_matching_search_str(search, exclude_inventory=True)
            else:
                obj = self.__get_nearest_object_matching_search_str(search, exclude_inventory=True)
            if obj is None:
                return False
        else:
            obj = self.__get_object_by_id(le.metadata["objects"], oid)
            if obj is False:
                return False

        # First, teleport the camera to the nearest navigable point to the object of interest.
        if self.navigation_graph is None:
            self.__generate_navigation_graph()
        nav_point_idx, face_obj_rot = self.__aim_camera_at_object(obj, self.object_target_camera_idx)

        # Get hotspots of the object from this vantage point.
        shown_obj_id = obj["objectId"]
        enc_obj_hotspots = self.get_hotspots(
            agent_id=None, camera_id=self.object_target_camera_idx, object_id=obj["objectId"]
        )

        parent_receptacles = self.get_parent_receptacles(obj, self.get_objects(self.controller.last_event))

        # Back off to container if object is fully occluded.
        if len(enc_obj_hotspots["hotspots"]) == 0:
            if parent_receptacles is not None and len(parent_receptacles) > 0:
                logger.warning('no hotspots for obj "%s", so checking parentReceptacles' % obj["objectId"])
                for receptacle_obj in parent_receptacles:
                    if "Floor" in receptacle_obj:  # ignore the floor as a parent since hotspotting it isn't helpful
                        continue
                    logger.info("... trying %s" % receptacle_obj)
                    shown_obj_id = receptacle_obj
                    enc_obj_hotspots = self.get_hotspots(
                        agent_id=None, camera_id=self.object_target_camera_idx, object_id=receptacle_obj
                    )
                    if len(enc_obj_hotspots["hotspots"]) == 0:
                        # Couldn't see receptacle, so recenter camera and get a new frame
                        nav_point_idx, face_obj_rot = self.__aim_camera_at_object(
                            le.get_object(receptacle_obj), self.object_target_camera_idx
                        )
                        enc_obj_hotspots = self.get_hotspots(
                            agent_id=None, camera_id=self.object_target_camera_idx, object_id=receptacle_obj
                        )
                        if len(enc_obj_hotspots["hotspots"]) == 0:
                            # Put camera back on target object.
                            nav_point_idx, face_obj_rot = self.__aim_camera_at_object(
                                obj, self.object_target_camera_idx
                            )
                    if len(enc_obj_hotspots["hotspots"]) > 0:
                        break  # got a hotspot view for this parent
            if len(enc_obj_hotspots["hotspots"]) == 0:
                logger.warning(
                    'no hotspots for obj "%s", and no parentReceptacles hotspots,' % obj["objectId"]
                    + "so getting hotspots for nearest receptacle..."
                )
                nn_objs = [obj["objectId"]]
                while len(nn_objs) < 6:  # try limited number of nearby objects
                    nn_obj = self.__get_object_by_position(
                        le.metadata["objects"], obj["position"], ignore_object_ids=nn_objs
                    )
                    logger.info("... trying %s" % nn_obj["objectId"])
                    if nn_obj["receptacle"]:
                        if "Floor" not in nn_obj["objectId"]:  # ignore the floor as a parent
                            shown_obj_id = nn_obj["objectId"]
                            enc_obj_hotspots = self.get_hotspots(
                                agent_id=None, camera_id=self.object_target_camera_idx, object_id=nn_obj["objectId"]
                            )
                            if len(enc_obj_hotspots["hotspots"]) == 0:
                                # Couldn't see receptacle, so recenter camera and get a new frame
                                nav_point_idx, face_obj_rot = self.__aim_camera_at_object(
                                    nn_obj, self.object_target_camera_idx
                                )
                                enc_obj_hotspots = self.get_hotspots(
                                    agent_id=None, camera_id=self.object_target_camera_idx, object_id=nn_obj["objectId"]
                                )
                                if len(enc_obj_hotspots["hotspots"]) == 0:
                                    # Put camera back on target object.
                                    nav_point_idx, face_obj_rot = self.__aim_camera_at_object(
                                        obj, self.object_target_camera_idx
                                    )
                            if len(enc_obj_hotspots["hotspots"]) > 0:
                                break  # got a hotspot view for this candidate receptacle

                    nn_objs.append(nn_obj["objectId"])
        # If no receptacle hotspots can be found at all, just return the frame looking "at" the object.
        if len(enc_obj_hotspots["hotspots"]) == 0:
            logger.warning("no hotspots for parentReceptacles %s" % parent_receptacles)
            shown_obj_id = ""
        # Prep metadata to be sent up for UI.
        obj_view_pos_norm = self.__get_click_normalized_position_from_xz(
            self.navigation_points[nav_point_idx]["x"], self.navigation_points[nav_point_idx]["z"]
        )
        obj_data = {
            "success": True,
            "oid": obj["objectId"],  # the object matching the query
            "shown_oid": shown_obj_id,  # The object whose hotspots are shown
            "view_pos_norm": obj_view_pos_norm,  # Location of the viewing camera on the topdown map
            "view_rot_norm": [face_obj_rot[0], -face_obj_rot[1]],  # flip y from thor coords
            "pos_norm": self.__get_click_normalized_position_from_xz(obj["position"]["x"], obj["position"]["z"]),
        }
        obj_data.update({"view_%s" % k: enc_obj_hotspots[k] for k in enc_obj_hotspots})  # hotspot array and width data

        return obj_data

    def encode_image(self, img):
        return super().encode_image(img)

    def get_parent_receptacles(self, obj, objects):
        """
        Recursively traces custom properties that track where objects were placed to identify receptacles of an object
        when AI2-THOR's property parentReceptacles fails
        :param obj: The object whose receptacles need to be identified
        :param objects: Output of get_objects()
        """
        if "parentReceptacles" in obj and obj["parentReceptacles"] is not None:
            return obj["parentReceptacles"]

        elif "simbotLastParentReceptacle" in obj:
            immediate_parent_receptacle = obj["simbotLastParentReceptacle"]
            if immediate_parent_receptacle is not None and immediate_parent_receptacle != obj["objectId"]:
                # Second clause is to prevent infinite recursion in weird corner cases that should ideally never happen
                parent_receptacles = [immediate_parent_receptacle]
                immediate_parent_receptacle_obj = self.__get_object_by_id(objects, immediate_parent_receptacle)
                if type(immediate_parent_receptacle_obj) == dict:
                    further_parent_receptacles = self.get_parent_receptacles(immediate_parent_receptacle_obj, objects)
                    if further_parent_receptacles is not None:
                        parent_receptacles += further_parent_receptacles
                return parent_receptacles

        return None

    def success(self):
        """
        When an episode ends, the parent function of done() will call this to see whether the episode can stop.
        """
        return True  # with the THOR backend, we can just say go ahead and stop

    def __get_agent_poses(self):
        """
        Return current poses of agents
        """
        if self.controller is None:
            return None
        if self.commander_embodied:
            cmd_xy = self.__get_agent_click_normalized_position(agent_id=0)
            cmd_r = self.__get_agent_click_rotation(agent_id=0)
            dri_xy = self.__get_agent_click_normalized_position(agent_id=1)
            dri_r = self.__get_agent_click_rotation(agent_id=1)
            return [(cmd_xy[0], cmd_xy[1], cmd_r[0], cmd_r[1]), (dri_xy[0], dri_xy[1], dri_r[0], dri_r[1])]
        else:
            e = self.controller.last_event
            cmd_xy = self.__get_agent_click_normalized_position(agent_metadata=e.metadata["thirdPartyCameras"][0])
            cmd_r = self.__get_agent_click_rotation(agent_metadata=e.metadata["thirdPartyCameras"][0])
            dri_xy = self.__get_agent_click_normalized_position()
            dri_r = self.__get_agent_click_rotation()
        return [(cmd_xy[0], cmd_xy[1], cmd_r[0], cmd_r[1]), (dri_xy[0], dri_xy[1], dri_r[0], dri_r[1])]

    def __get_nav_graph_point(self, thor_x, thor_z, exclude_points=None, get_closest=True):
        """
        Get the index in the navigation graph nearest to the given x,z coord in AI2-THOR coordinates
        :param thor_x: x coordinate on AI2-THOR floor plan
        :param thor_z: z coordinate on AI2-THOR floor plan
        :param exclude_points: Any navigation graph points that cannot be used
        :param get_closest: If false, instead of returning closest navigation graph point, do nucleus sampling around
        the coordinate; if True return the closest navigation graph point
        """
        if self.navigation_graph is None:
            self.__generate_navigation_graph()
        t_point = nearest_t_d = None
        distances = []
        for idx in range(len(self.navigation_points)):
            if exclude_points is not None and idx in exclude_points:
                distances.append(float("inf"))
                continue
            d = np.abs(self.navigation_points[idx]["x"] - thor_x) + np.abs(self.navigation_points[idx]["z"] - thor_z)
            distances.append(d)
            if t_point is None or d < nearest_t_d:
                t_point = idx
                nearest_t_d = d
        if not get_closest:  # rather than returning closest point, do nucleus sampling on softmax of 1/d
            scores = [np.exp(1.0 / d) for d in distances]
            dps = {idx: scores[idx] / sum(scores) for idx in range(len(scores))}
            dnps = {}
            nucleus_density = 0.1
            nucleus_sum = 0
            for k, v in sorted(dps.items(), key=lambda item: item[1], reverse=True):
                dnps[k] = v if nucleus_sum < nucleus_density or len(dnps) == 0 else 0
                nucleus_sum += v
            nps = [dnps[idx] for idx in range(len(scores))]
            nps = [p / sum(nps) for p in nps]
            t_point = np.random.choice(list(range(len(self.navigation_points))), p=nps)
        return t_point

    def __get_nav_graph_rot(self, thor_x, thor_z, thor_facing_x, thor_facing_z):
        """
        Get the cardinal direction to rotate to, to be facing (thor_facing_x, thor_facing_x=z) when standing at
        (thor_x, thor_z)
        :param thor_x: x Coordinate on Ai2-THOR floor plan where agent is standing
        :param thor_z: z Coordinate on Ai2-THOR floor plan where agent is standing
        :param thor_facing_x: x Coordinate on Ai2-THOR floor plan where agent is desired to face
        :param thor_facing_z: z Coordinate on Ai2-THOR floor plan where agent is desired to face
        """
        # Determine target rotation.
        if np.abs(thor_x - thor_facing_x) > np.abs(thor_z - thor_facing_z):  # Difference is greater in the x direction.
            if thor_x - thor_facing_x > 0:  # Destination to the x left
                t_rot = (-1, 0)
            else:
                t_rot = (1, 0)
        else:  # Difference is greater in the z direction
            if thor_z - thor_facing_z > 0:  # Destination to the z above
                t_rot = (0, -1)
            else:
                t_rot = (0, 1)

        return t_rot

    def __generate_navigation_graph(self):
        """
        Generate navigation graph: We construct a directed graph with nodes representing agent
        position and rotation. For every occupiable grid point on the map, we create four nodes for each orientation.
        Orientation nodes at a single occupiable point are connected with directed edges for turns.
        Occupiable positions are connected with directed edges that preserve orientation.
        """
        if debug_print_all_sim_steps:
            logger.info("step %s", "GetReachablePositions")
        event = self.controller.step(action="GetReachablePositions")
        p = event.metadata["actionReturn"]

        ng = nx.DiGraph()
        rotations = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        for idx in range(len(p)):
            for rx, rz in rotations:
                ng.add_node((idx, rx, rz))
        for idx in range(len(p)):
            for irx, irz in rotations:
                for jrx, jrz in rotations:
                    if irx + jrx == 0 or irz + jrz == 0:
                        continue  # antipodal or identical
                    ng.add_edge((idx, irx, irz), (idx, jrx, jrz))
            for jdx in range(len(p)):
                if idx == jdx:
                    continue
                rx = rz = None
                if np.isclose(p[idx]["z"] - p[jdx]["z"], 0):
                    if np.isclose(p[idx]["x"] - p[jdx]["x"], self.grid_size):
                        rx = -1
                        rz = 0
                    elif np.isclose(p[idx]["x"] - p[jdx]["x"], -self.grid_size):
                        rx = 1
                        rz = 0
                elif np.isclose(p[idx]["x"] - p[jdx]["x"], 0):
                    if np.isclose(p[idx]["z"] - p[jdx]["z"], self.grid_size):
                        rx = 0
                        rz = -1
                    elif np.isclose(p[idx]["z"] - p[jdx]["z"], -self.grid_size):
                        rx = 0
                        rz = 1
                if rx is not None and rz is not None:
                    ng.add_edge((idx, rx, rz), (jdx, rx, rz))
        self.navigation_graph = ng
        self.navigation_points = p

    def __update_custom_property_cooked(self, event):
        """
        Check whether objects are cooked and update custom property. Augments objects marked as cooked according to the
        AI2-THOR property "cooked" by using get_parent_receptacles() to track if any objects were newly placed on a
        hot burner or in an switched on microwave
        """
        cur_event_objects = self.get_objects(event)
        # Mark all objects detected by THOR as cooked
        thor_cooked = [obj for obj in cur_event_objects if obj["isCooked"]]
        for obj in thor_cooked:
            self.__update_custom_object_metadata(obj["objectId"], "simbotIsCooked", True)
        candidate_objs = [
            obj
            for obj in cur_event_objects
            if obj["cookable"] and not obj["isCooked"] and ("simbotIsCooked" not in obj or not obj["simbotIsCooked"])
        ]
        for cur_obj in candidate_objs:
            parent_receptacle_ids = self.get_parent_receptacles(cur_obj, cur_event_objects)
            if parent_receptacle_ids is not None and len(parent_receptacle_ids) > 0:
                parent_receptacle_ids = set(parent_receptacle_ids)
                parent_microwaves_on = [
                    obj["isToggled"]
                    for obj in cur_event_objects
                    if obj["objectId"] in parent_receptacle_ids and obj["objectType"] == "Microwave"
                ]
                if np.any(parent_microwaves_on):
                    self.__update_custom_object_metadata(cur_obj["objectId"], "simbotIsCooked", True)
                    continue

                burners = [
                    obj
                    for obj in cur_event_objects
                    if obj["objectId"] in parent_receptacle_ids and obj["objectType"] == "StoveBurner"
                ]
                # Depending on ai2thor version need to check either ObjectTemperature or temperature
                parent_burners_hot = list()
                for burner in burners:
                    if "ObjectTemperature" in burner and "Hot" in burner["ObjectTemperature"]:
                        parent_burners_hot.append(True)
                    elif "temperature" in burner and "Hot" in burner["temperature"]:
                        parent_burners_hot.append(True)
                    else:
                        parent_burners_hot.append(False)
                if np.any(parent_burners_hot):
                    self.__update_custom_object_metadata(cur_obj["objectId"], "simbotIsCooked", True)

    def __update_custom_property_boiled(self, last_event_objects, event):
        """
        Check whether objects are boiled and update custom property. An object is considered boiled if it just got
        cooked in the last time step, and was in a container filled with liquid at this time
        """
        cur_event_objects = self.get_objects(event)

        # Find objects whose isCooked property flipped after the last action
        just_got_cooked = [
            obj
            for obj in cur_event_objects
            if obj["isCooked"]
            and (
                last_event_objects is None
                or type(self.__get_object_by_id(last_event_objects, obj["objectId"])) != dict
                or not self.__get_object_by_id(last_event_objects, obj["objectId"])["isCooked"]
            )
        ]
        for obj in just_got_cooked:
            parent_receptacles = self.get_parent_receptacles(obj, cur_event_objects)
            if parent_receptacles is not None and last_event_objects is not None:
                for parent_receptacle_id in parent_receptacles:
                    parent_receptacle = self.__get_object_by_id(cur_event_objects, parent_receptacle_id)
                    if type(parent_receptacle) == dict and (
                        parent_receptacle["isFilledWithLiquid"]
                        or (
                            "simbotIsFilledWithWater" in parent_receptacle
                            and parent_receptacle["simbotIsFilledWithWater"]
                        )
                    ):
                        self.__update_custom_object_metadata(obj["objectId"], "simbotIsBoiled", True)
                        break

    def __get_oid_at_frame_xy_with_affordance(
        self,
        x,
        y,
        le,
        sim_agent_id,
        candidate_affordance_properties,
        region_backoff=False,
        region_radius=None,
        allow_agent_as_target=False,
    ):
        """
        Identify an object around relative coordinate (x, y) in the egocentric frame that satisfies required affordances
        :param x: Relative coordinate x in [0, 1) at which to find an object from the segmentation frame
        :param y: Relative coordinate y in [0, 1) at which to find an object from the segmentation frame
        :param le: the last event from the simulator
        :param sim_agent_id: 0 for Commander and 1 for Driver/ Follower
        :param candidate_affordance_properties: a list of dictionaries, an object's metadata must match key-value pairs
        in one of these to be returned; see values in self.action_to_affordances for examples
        :param region_backoff: if True and (x, y) gives no oid or an oid lacking the given affordance, do a radial
        search in a region around (x, y) using IOU between the region and objects
        :param allow_agent_as_target: if True, allow object ids starting with `agent_` to be returned as valid if
        they're at the EXACT (x, y) click position only; will return oid `agent_[bodypart]` and obj None; False for
        TEACh dataset
        """
        assert not region_backoff or region_radius is not None
        interacted_oid = interacted_obj = None

        # if last event doesn't have a segmentation frame, get one
        if le.instance_segmentation_frame is None:
            if debug_print_all_sim_steps:
                logger.info("step %s", dict(action="Pass", agentId=sim_agent_id, renderObjectImage=True))
            self.controller.step(action="Pass", agentId=sim_agent_id, renderObjectImage=True)
            le = (
                self.controller.last_event.events[sim_agent_id]
                if self.commander_embodied
                else self.controller.last_event
            )

        # Check if we can get a matching object at exactly (x, y)
        instance_segs = np.array(le.instance_segmentation_frame)
        color_to_object_id = le.color_to_object_id
        pixel_x, pixel_y = int(np.round(x * self.web_window_size)), int(np.round(y * self.web_window_size))
        instance_color_id = tuple(instance_segs[pixel_y, pixel_x])
        xy_match = False
        if instance_color_id in color_to_object_id:
            oid = color_to_object_id[instance_color_id]
            if allow_agent_as_target and oid[: len("agent_")] == "agent_":
                return oid, None
            if oid in le.instance_detections2D:
                obj = self.__get_object_by_id(self.get_objects(self.controller.last_event), oid)
                if obj:
                    for affordance_properties in candidate_affordance_properties:
                        if np.all([k in obj and obj[k] == affordance_properties[k] for k in affordance_properties]):
                            interacted_oid = oid
                            interacted_obj = obj
                            xy_match = True
                            break

        # Do a radial search in a region around (x, y)
        if not xy_match and region_backoff:
            # Count pixels of affordance-matching objects in the interaction region.
            affordance_matching_oid_pixel_counts = {}
            affordance_matching_oid_total_pixels = {}
            affordance_matching_oid_to_obj = {}
            affordance_nonmatching_oids = set()
            for rx in range(max(0, pixel_x - region_radius), min(self.web_window_size, pixel_x + region_radius)):
                for ry in range(max(0, pixel_y - region_radius), min(self.web_window_size, pixel_y + region_radius)):
                    instance_color_id = tuple(instance_segs[ry, rx])
                    if instance_color_id in color_to_object_id:
                        oid = color_to_object_id[instance_color_id]
                        if oid in affordance_nonmatching_oids:  # seen oid, obj metadata does not match affordances
                            continue
                        if oid in affordance_matching_oid_pixel_counts:  # seen oid with matching affordance
                            affordance_matching_oid_pixel_counts[oid] += 1
                        else:  # Unseen oid, so find obj and check affordances
                            if oid in le.instance_detections2D:
                                obj = self.__get_object_by_id(self.get_objects(self.controller.last_event), oid)
                                obj_affordance_match = False
                                if obj:
                                    for affordance_properties in candidate_affordance_properties:
                                        if np.all(
                                            [
                                                k in obj and obj[k] == affordance_properties[k]
                                                for k in affordance_properties
                                            ]
                                        ):
                                            affordance_matching_oid_pixel_counts[oid] = 1
                                            affordance_matching_oid_to_obj[oid] = obj
                                            # Get the total pixel count for this object's mask in the frame.
                                            affordance_matching_oid_total_pixels[oid] = np.sum(
                                                np.all(instance_segs == instance_color_id, axis=2).astype(np.uint8)
                                            )
                                            obj_affordance_match = True
                                            break

                                    if not obj_affordance_match:
                                        affordance_nonmatching_oids.add(oid)

            # Tiebreak using IOU
            if len(affordance_matching_oid_pixel_counts) > 0:
                oid_ious = {
                    oid: affordance_matching_oid_pixel_counts[oid] / affordance_matching_oid_total_pixels[oid]
                    for oid in affordance_matching_oid_pixel_counts
                }
                oid_ious_s = sorted(oid_ious.items(), key=lambda k: k[1], reverse=True)
                interacted_oid = oid_ious_s[0][0]
                interacted_obj = affordance_matching_oid_to_obj[interacted_oid]

        return interacted_oid, interacted_obj

    def add_interaction(self, interaction, on_oid=None, force=False):
        """
        Execute an Interaction - a formatted action - and add it to the current episode
        :param interaction: instance of class Interaction defined in dataset.py
        :param on_oid: To be used only during replay; allows forcing an action to take place on a specific object
        :param force: To be used only during replay; force the action to be successful even if the agent is not near
        enough to it
        """
        if on_oid is not None:
            logger.info("SimulatorTHOR add_interaction invoked with an on_oid; disallowed outside replay scripts")
        if self.controller is None:
            message = "Simulator was not initialized. Possible resolution: Start new episode."
            self.logger.warning(message)
            raise Exception(message)

        sim_agent_id = interaction.agent_id if self.commander_embodied else 0
        le = self.controller.last_event.events[sim_agent_id] if self.commander_embodied else self.controller.last_event
        objects_before_cur_event = copy.deepcopy(self.get_objects(le))

        action_definition = self._dataset.definitions.map_actions_id2info[interaction.action.action_id]
        action_type = action_definition["action_type"]
        action_name = action_definition["action_name"]
        pose_delta = action_definition["pose_delta"]

        if interaction.action.action_type == "Motion":

            if not self.commander_embodied and interaction.agent_id == 0:  # Commander third party camera motion
                event = self.controller.last_event
                current_position = event.metadata["thirdPartyCameras"][0]["position"]
                current_rotation = event.metadata["thirdPartyCameras"][0]["rotation"]

                new_position = current_position
                new_rotation = current_rotation

                # Get a rotation unit vector in the direction the camera is facing along the xz-plane.
                unit_rot = self.__get_xz_rot_from_y(current_rotation["y"])

                # Implement each movement as a function of current rotation direction.
                if action_name == "Forward":
                    new_position["x"] += self.grid_size * unit_rot[0]
                    new_position["z"] += self.grid_size * unit_rot[1]
                elif action_name == "Backward":
                    new_position["x"] += self.grid_size * -unit_rot[0]
                    new_position["z"] += self.grid_size * -unit_rot[1]
                elif action_name == "Turn Left":
                    new_rotation["y"] = (new_rotation["y"] - 90) % 360
                elif action_name == "Turn Right":
                    new_rotation["y"] = (new_rotation["y"] + 90) % 360
                elif action_name == "Look Up":  # strafe
                    new_position["y"] += self.grid_size
                    pass
                elif action_name == "Look Down":  # strafe
                    new_position["y"] -= self.grid_size
                    pass
                elif action_name == "Pan Left":  # strafe
                    rot_facing_left = self.__get_xz_rot_from_y((current_rotation["y"] - 90) % 360)
                    new_position["x"] += self.grid_size * rot_facing_left[0]
                    new_position["z"] += self.grid_size * rot_facing_left[1]
                elif action_name == "Pan Right":  # strafe
                    rot_facing_right = self.__get_xz_rot_from_y((current_rotation["y"] + 90) % 360)
                    new_position["x"] += self.grid_size * rot_facing_right[0]
                    new_position["z"] += self.grid_size * rot_facing_right[1]
                elif action_name == "Stop":
                    pass
                else:
                    logger.warning("%s: Motion not supported" % action_name)
                    interaction.action.success = 0
                    return False, "", None

                tpc_ac = dict(
                    action="UpdateThirdPartyCamera", thirdPartyCameraId=0, rotation=new_rotation, position=new_position
                )
                if debug_print_all_sim_steps:
                    logger.info("step %s", tpc_ac)
                self.controller.step(tpc_ac)
                event = self.controller.last_event

                super().add_interaction(interaction)
                if event.metadata["lastActionSuccess"]:
                    interaction.action.success = 1
                    return True, "", None
                else:
                    interaction.action.success = 0
                    return (
                        event.metadata["lastActionSuccess"],
                        event.metadata["errorMessage"],
                        self.__thor_error_to_help_message(event.metadata["errorMessage"]),
                    )

            else:  # Agent motion

                # Note on events returned with multiagent: accessing e metadata directly keys into the event
                # corresponding to the agent who just took the action, so logic like that below does not need any
                # special hooks for specifying the agent id.
                if action_name == "Forward":
                    ac = dict(action="MoveAhead", agentId=sim_agent_id, moveMagnitude=pose_delta[0], forceAction=True)
                    if debug_print_all_sim_steps:
                        logger.info("step %s", ac)
                    e = self.controller.step(ac)
                elif action_name == "Backward":
                    ac = dict(action="MoveBack", agentId=sim_agent_id, moveMagnitude=-pose_delta[0], forceAction=True)
                    if debug_print_all_sim_steps:
                        logger.info("step %s", ac)
                    e = self.controller.step(ac)
                elif action_name == "Look Up":
                    ac = dict(action="LookUp", agentId=sim_agent_id, degrees=-pose_delta[4], forceAction=True)
                    if debug_print_all_sim_steps:
                        logger.info("step %s", ac)
                    e = self.controller.step(ac)
                elif action_name == "Look Down":
                    ac = dict(action="LookDown", agentId=sim_agent_id, degrees=pose_delta[4], forceAction=True)
                    if debug_print_all_sim_steps:
                        logger.info("step %s", ac)
                    e = self.controller.step(ac)
                elif action_name == "Turn Left":
                    ac = dict(action="RotateLeft", agentId=sim_agent_id, degrees=pose_delta[5], forceAction=True)
                    if debug_print_all_sim_steps:
                        logger.info("step %s", ac)
                    e = self.controller.step(ac)
                elif action_name == "Turn Right":
                    ac = dict(action="RotateRight", agentId=sim_agent_id, degrees=-pose_delta[5], forceAction=True)
                    if debug_print_all_sim_steps:
                        logger.info("step %s", ac)
                    e = self.controller.step(ac)
                elif action_name == "Pan Left":  # strafe left
                    ac = dict(action="MoveLeft", agentId=sim_agent_id, forceAction=True)
                    if debug_print_all_sim_steps:
                        logger.info("step %s", ac)
                    e = self.controller.step(ac)
                elif action_name == "Pan Right":  # strafe right
                    ac = dict(action="MoveRight", agentId=sim_agent_id, forceAction=True)
                    if debug_print_all_sim_steps:
                        logger.info("step %s", ac)
                    e = self.controller.step(ac)
                elif action_name == "Stop":  # do nothing
                    ac = dict(action="Pass", agentId=sim_agent_id)
                    if debug_print_all_sim_steps:
                        logger.info("step %s", ac)
                    e = self.controller.step(ac)
                else:
                    logger.warning("%s: Motion not supported" % action_name)
                    interaction.action.success = 0
                    return False, "", None

                # Pose returned should be the one for the agent who just took an action based on behavior of event
                # returns.
                interaction.action.pose = self.get_current_pose(agent_id=sim_agent_id)
                super().add_interaction(interaction)

                # Return action success data.
                if e.metadata["lastActionSuccess"]:
                    interaction.action.success = 1
                    return True, "", None
                else:
                    interaction.action.success = 0
                    return (
                        e.metadata["lastActionSuccess"],
                        e.metadata["errorMessage"],
                        self.__thor_error_to_help_message(e.metadata["errorMessage"]),
                    )

        elif action_type == "MapGoal":
            if action_name == "Navigation":

                # Get the latest reachable positions for this scene and build navigation graph.
                if self.navigation_graph is None:
                    self.__generate_navigation_graph()

                # Determine target grid cell based on click (x, y).
                graph_constrained = True  # whether to abide by navigation graph.
                if self.commander_embodied:
                    agent_data = self.controller.last_event.events[sim_agent_id].metadata["agent"]
                else:
                    if interaction.agent_id == 0:  # it's the floating camera
                        graph_constrained = False  # camera can fly over/through anything
                        agent_data = self.controller.last_event.metadata["thirdPartyCameras"][0]
                    else:  # it's the driver robot agent
                        agent_data = self.controller.last_event.metadata["agent"]
                # Topdown camera mapping derived from
                # https://github.com/allenai/ai2thor/issues/445#issuecomment-713916052
                # z is flipped from top-to-bottom y of UI, so 1 - y = z
                sx, sz = interaction.action.start_x, (1 - interaction.action.start_y)
                tx, tz = self.topdown_lower_left_xz + 2 * self.topdown_cam_orth_size * np.array((sx, sz))
                t_face_x, t_face_z = self.topdown_lower_left_xz + 2 * self.topdown_cam_orth_size * np.array(
                    (interaction.action.end_x, (1 - interaction.action.end_y))
                )
                t_rot = self.__get_nav_graph_rot(tx, tz, t_face_x, t_face_z)
                s_point = nearest_s_d = None
                if graph_constrained:  # Only need to find start graph node if graph constrained.
                    t_point = self.__get_nav_graph_point(tx, tz)
                    for idx in range(len(self.navigation_points)):
                        d = np.abs(self.navigation_points[idx]["x"] - agent_data["position"]["x"]) + np.abs(
                            self.navigation_points[idx]["z"] - agent_data["position"]["z"]
                        )
                        if s_point is None or d < nearest_s_d:
                            s_point = idx
                            nearest_s_d = d
                else:
                    t_point = None

                # Determine current rotation.
                s_rot = self.__get_xz_rot_from_y(agent_data["rotation"]["y"])
                if s_rot is None:
                    msg = "%.4f source rotation failed to align" % agent_data["rotation"]["y"]
                    logger.info(msg)
                    interaction.action.success = 0
                    return False, msg

                # Build shortest path and unpack actions needed to execute it.
                action_sequence = []
                lrots = [(-1, 0, 0, -1), (0, -1, 1, 0), (1, 0, 0, 1), (0, 1, -1, 0)]
                rrots = [(0, 1, 1, 0), (1, 0, 0, -1), (0, -1, -1, 0), (-1, 0, 0, 1)]
                if graph_constrained:
                    node_path = nx.shortest_path(
                        self.navigation_graph, (s_point, s_rot[0], s_rot[1]), (t_point, t_rot[0], t_rot[1])
                    )
                    # Decode action sequence from graph path.
                    for idx in range(len(node_path) - 1):
                        # Determine action to get from node idx to node idx + 1.
                        if node_path[idx][0] != node_path[idx + 1][0]:  # moving forward to a new node.
                            action_sequence.append("forward")  # use web UI names to facilitate feedback thru it.
                        else:
                            rot_trans = (
                                node_path[idx][1],
                                node_path[idx][2],
                                node_path[idx + 1][1],
                                node_path[idx + 1][2],
                            )
                            if rot_trans in lrots:  # rotate left
                                action_sequence.append("turn_left")
                            elif rot_trans in rrots:  # rotate right
                                action_sequence.append("turn_right")
                            else:
                                msg = "could not determine action from points:", node_path[idx], node_path[idx + 1]
                                logger.info(msg)
                                interaction.action.success = 0
                                return False, msg
                else:
                    # Create action sequence directly from source and target world coordinates.
                    # Without graph constraints, the camera will fly in fixed orientation to its destination (strafing)
                    # and then orient to target orientation once at the destination.
                    cx = agent_data["position"]["x"]
                    cz = agent_data["position"]["z"]
                    while tx - cx > self.grid_size:  # target is x right
                        action_sequence.append(
                            "forward"
                            if s_rot[0] == 1
                            else "backward"
                            if s_rot[0] == -1
                            else "pan_left"
                            if s_rot[1] == -1
                            else "pan_right"
                        )
                        cx += self.grid_size
                    while cx - tx > self.grid_size:  # target is x left
                        action_sequence.append(
                            "forward"
                            if s_rot[0] == -1
                            else "backward"
                            if s_rot[0] == 1
                            else "pan_left"
                            if s_rot[1] == 1
                            else "pan_right"
                        )
                        cx -= self.grid_size
                    while tz - cz > self.grid_size:  # target is z right
                        action_sequence.append(
                            "forward"
                            if s_rot[1] == 1
                            else "backward"
                            if s_rot[1] == -1
                            else "pan_left"
                            if s_rot[0] == 1
                            else "pan_right"
                        )
                        cz += self.grid_size
                    while cz - tz > self.grid_size:  # target is z left
                        action_sequence.append(
                            "forward"
                            if s_rot[1] == -1
                            else "backward"
                            if s_rot[1] == 1
                            else "pan_left"
                            if s_rot[0] == -1
                            else "pan_right"
                        )
                        cz -= self.grid_size
                    if s_rot != t_rot:
                        rot_trans = (s_rot[0], s_rot[1], t_rot[0], t_rot[1])
                        if rot_trans in lrots:  # just need to rotate left
                            action_sequence.append("turn_left")
                        elif rot_trans in rrots:  # just need to rotate right
                            action_sequence.append("turn_right")
                        else:  # need to turn around
                            action_sequence.extend(["turn_left", "turn_left"])

            else:
                msg = "%s: NavigationGoal not supported" % action_name
                logger.info(msg)
                interaction.action.success = 0
                return False, msg

            super().add_interaction(interaction)  # log successful nav action sequence initiated.

            # Create and return error and message structure.
            interaction.action.success = 1
            return True, action_sequence

        # Take the specified action on the target object at (x, y) on the screen.
        elif action_type == "ObjectInteraction":
            interacted_oid = None
            x, y = interaction.action.x, interaction.action.y
            msg = action = event = None

            # This is a list of possible dictionaries of affordance a valid target object for a manipulation needs to
            # have and is populated by the actual action we are attempting
            candidate_affordance_properties = list()

            # check if agent is holding knife in hand
            inventory_objects_before_action = self.get_inventory_objects(le)
            handoff = False  # whether we should bypass normal action taking and do a handoff instead

            if action_name == "Pickup":
                action = dict(action="PickupObject", agentId=sim_agent_id)
                candidate_affordance_properties = self.action_to_affordances["Pickup"]
            elif action_name == "Place":
                # Check whether holding anything.
                if len(inventory_objects_before_action) == 0:
                    event = None
                    msg = "%s: ObjectInteraction only supported when holding an object" % action_name
                else:
                    # Check whether the click position is the other agent, in which case we instead do a handoff.
                    if (
                        self.commander_embodied
                        and len(self.get_inventory_objects(self.controller.last_event.events[(sim_agent_id + 1) % 2]))
                        == 0
                    ):
                        interacted_oid, _ = self.__get_oid_at_frame_xy_with_affordance(
                            x, y, le, sim_agent_id, {}, allow_agent_as_target=True
                        )
                        if interacted_oid is not None and "agent_" in interacted_oid:
                            # Check that agent target is close enough for a handoff.
                            if (
                                self.__agent_dist_to_agent(sim_agent_id, (sim_agent_id + 1) % 2)
                                <= self.visibility_distance
                            ):
                                # Place the held object on the floor so other agent can pick it up.
                                floor_place = dict(
                                    action="PutObject", objectId=self.floor_oid, agentId=sim_agent_id, forceAction=True
                                )
                                if debug_print_all_sim_steps:
                                    logger.info("step %s", floor_place)
                                drop_e = self.controller.step(floor_place)
                                if drop_e.metadata["lastActionSuccess"]:
                                    handoff = True
                                    action = dict(
                                        action="PickupObject", agentId=(sim_agent_id + 1) % 2, forceAction=True
                                    )
                                    interacted_oid = inventory_objects_before_action[0]["objectId"]
                                else:
                                    msg = "You are unable to hand off the object to your partner."
                            else:
                                msg = "Your partner is too far away for a handoff."
                    if not handoff:
                        action = dict(action="PutObject", agentId=sim_agent_id, forceAction=True, placeStationary=True)
                        candidate_affordance_properties = self.action_to_affordances["Place"]
            elif action_name == "Open":
                action = dict(action="OpenObject", agentId=sim_agent_id)
                candidate_affordance_properties = self.action_to_affordances["Open"]
            elif action_name == "Close":
                action = dict(action="CloseObject", agentId=sim_agent_id)
                candidate_affordance_properties = self.action_to_affordances["Close"]
            elif action_name == "ToggleOn":
                action = dict(action="ToggleObjectOn", agentId=sim_agent_id)
                candidate_affordance_properties = self.action_to_affordances["ToggleOn"]
            elif action_name == "ToggleOff":
                action = dict(action="ToggleObjectOff", agentId=sim_agent_id)
                candidate_affordance_properties = self.action_to_affordances["ToggleOff"]
            elif action_name == "Slice":
                if (
                    len(inventory_objects_before_action) == 0
                    or "Knife" not in inventory_objects_before_action[0]["objectType"]
                ):
                    event = None
                    msg = "%s: ObjectInteraction only supported for held object Knife" % action_name
                else:
                    action = dict(action="SliceObject", agentId=sim_agent_id, x=x, y=y)
                    candidate_affordance_properties = self.action_to_affordances["Slice"]
            elif action_name == "Dirty":
                action = dict(action="DirtyObject", agentId=sim_agent_id)
                candidate_affordance_properties = self.action_to_affordances["Dirty"]
            elif action_name == "Clean":
                action = dict(action="CleanObject", agentId=sim_agent_id)
                candidate_affordance_properties = self.action_to_affordances["Clean"]
            elif action_name == "Fill":
                action = dict(action="FillObjectWithLiquid", agentId=sim_agent_id, fillLiquid="water")
                candidate_affordance_properties = self.action_to_affordances["Fill"]
            elif action_name == "Empty":
                action = dict(action="EmptyLiquidFromObject", agentId=sim_agent_id)
                candidate_affordance_properties = self.action_to_affordances["Empty"]
            elif action_name == "Pour":
                if len(inventory_objects_before_action) == 0:
                    event = None
                    msg = "%s: ObjectInteraction only supported for held object filled with liquid" % action_name
                else:
                    held_obj = self.__get_object_by_id(
                        self.get_objects(self.controller.last_event), inventory_objects_before_action[0]["objectId"]
                    )
                    if not held_obj["isFilledWithLiquid"]:
                        event = None
                        msg = "%s: ObjectInteraction only supported for held object filled with liquid" % action_name
                    else:
                        fillLiquid = (
                            "coffee"
                            if "simbotIsFilledWithCoffee" in held_obj and held_obj["simbotIsFilledWithCoffee"]
                            else "water"
                        )
                        action = dict(action="FillObjectWithLiquid", agentId=sim_agent_id, fillLiquid=fillLiquid)
                        candidate_affordance_properties = self.action_to_affordances["Pour"]
            elif action_name == "Break":
                action = dict(action="BreakObject", agentId=sim_agent_id)
                candidate_affordance_properties = self.action_to_affordances["Break"]
            else:
                event = None
                msg = "%s: ObjectInteraction not supported" % action_name

            if action is not None:  # action was recognized and dict is prepped, so get oid for interaction and step
                # Get interaction oid and associated object from the segmentation mask.
                if handoff:
                    interacted_obj = None
                else:
                    interacted_oid, interacted_obj = self.__get_oid_at_frame_xy_with_affordance(
                        x,
                        y,
                        le,
                        sim_agent_id,
                        candidate_affordance_properties,
                        region_backoff=True,
                        region_radius=self.hotspot_pixel_width,
                    )
                if interacted_oid is not None:
                    action["objectId"] = interacted_oid
                    if not interacted_obj and not handoff:
                        msg = "Could not retrieve object metadata for '%s'" % interacted_oid

                    # Need to do a manual visibilityDistance check because we're using forceAction=True to cause
                    # put into any receptacle regardless of metadata constraint (e.g., no sponge in microwave).
                    raycast_action = dict(action="GetCoordinateFromRaycast", x=x, y=y, agentId=sim_agent_id)
                    if debug_print_all_sim_steps:
                        logger.info("step %s", raycast_action)
                    raycast_e = self.controller.step(raycast_action)
                    clicked_xyz = raycast_e.metadata["actionReturn"]
                    if (
                        np.linalg.norm([clicked_xyz[c] - le.metadata["agent"]["position"][c] for c in ["x", "y", "z"]])
                        > self.visibility_distance
                    ):
                        msg = "%s is too far away to be interacted with" % interacted_oid
                        del action["objectId"]  # don't take the action because the obj is too far away.

                # Override objectId if specified
                if on_oid is not None:
                    action["objectId"] = on_oid

                if force:
                    action["forceAction"] = True

                # Actually take the simulator step.
                if "objectId" in action:

                    # If PutObject, add back in (x, y) so raycast is used to inform final position on target.
                    if action["action"] == "PutObject":
                        action["x"] = x
                        action["y"] = y
                        action["putNearXY"] = True
                        ac = {k: action[k] for k in action if k != "objectId"}
                        if debug_print_all_sim_steps:
                            logger.info("step %s", ac)
                        event = self.controller.step(ac)

                    # If we're about to slice an object held by the other agent, cancel the action.
                    # If slice happens with a held object, THOR doesn't de-register inventory.
                    # We tried DropHandObject, but then the slices scatter around the robot base and trap it.
                    elif (
                        self.commander_embodied
                        and action["action"] == "SliceObject"
                        and action["objectId"]
                        in [
                            obj["objectId"]
                            for obj in self.get_inventory_objects(
                                self.controller.last_event.events[(sim_agent_id + 1) % 2]
                            )
                        ]
                    ):
                        msg = "You cannot slice something while your partner is holding it."

                    # Else, just take the action we prepared already.
                    else:
                        if debug_print_all_sim_steps:
                            logger.info("step %s", action)
                        event = self.controller.step(action)

                    # If it is a pour action empty the inventory object
                    if action_name == "Pour":
                        interacted_obj = self.__get_object_by_id(
                            self.get_objects(self.controller.last_event), action["objectId"]
                        )
                        if event.metadata["lastActionSuccess"] or interacted_obj["objectType"] in [
                            "Sink",
                            "SinkBasin",
                            "Bathtub",
                            "BathtubBasin",
                        ]:
                            held_obj = self.__get_object_by_id(
                                self.get_objects(self.controller.last_event),
                                self.get_inventory_objects(self.controller.last_event)[0]["objectId"],
                            )
                            empty_action = dict(
                                action="EmptyLiquidFromObject", agentId=sim_agent_id, objectId=held_obj["objectId"]
                            )
                            if debug_print_all_sim_steps:
                                logger.info("step %s", empty_action)
                            event = self.controller.step(empty_action)
                            if event.metadata["lastActionSuccess"]:
                                self.__update_custom_object_metadata(
                                    held_obj["objectId"], "simbotIsFilledWithCoffee", False
                                )

                    # Set custom message for Pickup action on success.
                    # Note: action taken is actually the pickup by the partner agent on a handoff
                    if action_name == "Pickup" or (handoff and action_name == "Place"):
                        inventory_objects = self.get_inventory_objects(event)
                        if event is not None and event.metadata["lastActionSuccess"] and len(inventory_objects) > 0:
                            msg = "Picked up %s" % inventory_objects[0]["objectType"]

                            # Update parent/child relationships in inventory.
                            for obj in inventory_objects:
                                self.__update_custom_object_metadata(obj["objectId"], "simbotPickedUp", 1)
                                if "simbotLastParentReceptacle" in self.__custom_object_metadata[obj["objectId"]]:
                                    parent_receptacle = self.__custom_object_metadata[obj["objectId"]][
                                        "simbotLastParentReceptacle"
                                    ]
                                    self.__delete_from_custom_object_metadata_list(
                                        parent_receptacle, "simbotIsReceptacleOf", obj["objectId"]
                                    )
                                self.__update_custom_object_metadata(
                                    obj["objectId"], "simbotLastParentReceptacle", None
                                )

                    elif action_name == "Place":
                        if event is not None and "objectId" in action and event.metadata["lastActionSuccess"]:
                            msg = "Placed in %s" % action["objectId"]
                            for obj in inventory_objects_before_action:
                                self.__update_custom_object_metadata(
                                    obj["objectId"], "simbotLastParentReceptacle", action["objectId"]
                                )
                                self.__append_to_custom_object_metadata_list(
                                    action["objectId"], "simbotIsReceptacleOf", obj["objectId"]
                                )

                elif msg is None:
                    msg = "Could not find a target object at the specified location"

            super().add_interaction(interaction)  # log attempt, regardless of success.

            # If the event succeeded, do manual simulation updates based on fixed state change rules.
            if event is not None and event.metadata["lastActionSuccess"]:
                if action["action"] == "SliceObject":
                    self.__transfer_custom_metadata_on_slicing_cracking(self.get_objects(event))

            # Update custom properties in case actions changed things up.
            self.__check_per_step_custom_properties(objects_before_cur_event)

            # If the event was created and succeeded, return with default msg.
            if event is not None and event.metadata["lastActionSuccess"]:
                interaction.action.success = 1
                # if we successfully interacted, we need to set with what oid
                assert interacted_oid is not None or on_oid is not None
                interaction.action.oid = interacted_oid
                return True, "%s @ (%.2f, %.2f)" % (action_name, x, y) if msg is None else msg, None
            else:
                interaction.action.success = 0
                if event is None:  # If the event call never even got made, use custom message.
                    return False, msg, self.__thor_error_to_help_message(msg)
                else:  # If the event call failed, use error from AI2THOR and try to generate human-readable system msg
                    if event.metadata["errorMessage"]:
                        return (
                            False,
                            event.metadata["errorMessage"],
                            self.__thor_error_to_help_message(event.metadata["errorMessage"]),
                        )
                    elif msg is not None:
                        return False, msg, self.__thor_error_to_help_message(msg)
                    else:
                        logger.warning(
                            "action was taken that failed but produced no custom or system error message: %s", action
                        )
                        return False, "", None

        elif action_type == "ChangeCamera":
            if not self.commander_embodied and interaction.agent_id == 0:
                interaction.action.success = 0
                return False, "Floating camera cannot perform a ChangeCamera action"

            if action_name == "BehindAboveOn":
                interaction.action.success = 0
                raise NotImplementedError("CameraChange functions are being phased out")
            elif action_name == "BehindAboveOff":
                interaction.action.success = 0
                raise NotImplementedError("CameraChange functions are being phased out")

            return  # noqa R502

        elif action_type == "ProgressCheck":
            # ProgressCheck actions are handled via calls made directly from simulator_base.
            super().add_interaction(interaction)
            return  # noqa R502

        elif action_type == "Keyboard":
            if interaction.agent_id == 0:  # Commander
                self.logger.debug("*** Commander - Keyboard: %s ***" % interaction.action.utterance)
            else:
                self.logger.debug("*** Driver - Keyboard: %s ***" % interaction.action.utterance)
            super().add_interaction(interaction)
            interaction.action.success = 1
            return  # noqa R502
        elif action_type == "Audio":
            if interaction.agent_id == 0:  # Commander
                self.logger.info("*** Commander - Audio: %s ***" % interaction.action.utterance)
            else:
                self.logger.info("*** Driver - Audio: %s ***" % interaction.action.utterance)
            super().add_interaction(interaction)
            interaction.action.success = 1
            return  # noqa R502
        else:
            logger.warning("%s: Not supported" % interaction.action.action_type)
            interaction.action.success = 0
            return  # noqa R502

    def __update_custom_coffee_prop(self, event, objs_before_event=None):
        """
        Check whether coffee has been made and update custom property - this uses get_parent_receptacles() for extra
        reliability and checks that a container just got placed in a coffee maker and the coffee maker was on
        """
        cur_objects = self.get_objects(event)
        coffee_maker_ids = set(
            [obj["objectId"] for obj in cur_objects if "CoffeeMachine" in obj["objectType"] and obj["isToggled"]]
        )
        for obj in cur_objects:
            prev_filled_with_liquid = False
            if objs_before_event is not None:
                prev_state = self.__get_object_by_id(objs_before_event, obj["objectId"])
                if prev_state:
                    prev_filled_with_liquid = prev_state["isFilledWithLiquid"]
            parent_receptacles = self.get_parent_receptacles(obj, cur_objects)
            placed_in_toggled_coffee_maker = False
            if parent_receptacles is not None and len(set(parent_receptacles).intersection(coffee_maker_ids)) > 0:
                placed_in_toggled_coffee_maker = True
            if (
                placed_in_toggled_coffee_maker
                and obj["canFillWithLiquid"]
                and obj["isFilledWithLiquid"]
                and not prev_filled_with_liquid
            ):
                self.__update_custom_object_metadata(obj["objectId"], "simbotIsFilledWithCoffee", True)

    def __update_sink_interaction_outcomes(self, event):
        """
        Force sink behaviour to be deterministic - if a faucet is turned on, clean all objects in the sink and
        fill objects that can be filled with water
        """
        cur_objects = self.get_objects(event)
        sink_objects = list()
        for obj in cur_objects:
            # Check if any sink basin is filled with water and clean all dirty objects in.
            if (
                "SinkBasin" in obj["objectType"]
                or "Sink" in obj["objectType"]
                or "BathtubBasin" in obj["objectType"]
                or "Bathtub" in obj["objectType"]
            ):
                # Fetch the faucet near the sink
                faucet_obj = self.__get_object_by_position(self.get_objects(event), obj["position"], obj_type="Faucet")
                if faucet_obj["isToggled"]:
                    sink_objects.append(obj)
        sink_obj_ids = set([obj["objectId"] for obj in sink_objects])
        objs_in_sink = list()
        for obj in cur_objects:
            parent_receptacles = self.get_parent_receptacles(obj, cur_objects)
            if parent_receptacles is not None:
                if len(set(parent_receptacles).intersection(sink_obj_ids)) > 0:
                    objs_in_sink.append(obj)

        for child_obj in objs_in_sink:
            if child_obj["isDirty"]:
                ac = dict(action="CleanObject", objectId=child_obj["objectId"], forceAction=True)
                if debug_print_all_sim_steps:
                    logger.info("step %s", ac)
                self.controller.step(ac)

            if child_obj["canFillWithLiquid"]:
                ac = dict(
                    action="FillObjectWithLiquid", objectId=child_obj["objectId"], fillLiquid="water", forceAction=True
                )
                if debug_print_all_sim_steps:
                    logger.info("step %s", ac)
                self.controller.step(ac)
                self.__update_custom_object_metadata(child_obj["objectId"], "simbotIsFilledWithWater", 1)

    def __thor_error_to_help_message(self, msg):
        """
        Translate AI2-THOR errorMessage field into something that can be shown as prompts to annotators for TEACh data
        collection
        """
        # Example: "Floor|+00.00|+00.00|+00.00 must have the property CanPickup to be picked up." # noqa: E800
        if "CanPickup to be" in msg:
            return 'Object "%s" can\'t be picked up.' % msg.split()[0].split("|")[0]
        # Example: "Object ID appears to be invalid." # noqa: E800
        if ("Object ID" in msg and "invalid" in msg) or "Could not retrieve object" in msg:
            return "Could not determine what object was clicked."
        # Example "Can't place an object if Agent isn't holding anything # noqa: E800
        if "if Agent isn't holding" in msg:
            return "Must be holding an object first."
        # Example: "Slice: ObjectInteraction only supported for held object Knife" # noqa: E800
        if "Slice: ObjectInteraction" in msg:
            return "Must be holding a knife."
        # Example: "object is not toggleable" # noqa: E800
        if "not toggleable" in msg:
            return "Object cannot be turned on or off."
        # Example: "can't toggle object off if it's already off!" # noqa: E800
        if "toggle object off if" in msg:
            return "Object is already turned off."
        # Example: "can't toggle object on if it's already on!" # noqa: E800
        if "toggle object on if" in msg:
            return "Object is already turned on."
        # Example: "CounterTop|-00.08|+01.15|00.00 is not an Openable object" # noqa: E800
        if "is not an Openable object" in msg:
            return 'Object "%s" can\'t be opened.' % msg.split()[0].split("|")[0]
        # Example: "CounterTop_d7cc8dfe Does not have the CanBeSliced property!" # noqa: E800
        if "Does not have the CanBeSliced" in msg:
            return "Object cannot be sliced."
        # Example: "Object failed to open/close successfully." # noqa: E800
        if "failed to open/close" in msg:
            return "Something is blocking the object from opening or closing. Move farther away or remove obstruction."
        # Example: "StandardIslandHeight is blocking Agent 0 from moving 0" # noqa: E800
        if "is blocking" in msg:
            return "Something is blocking the robot from moving in that direction."
        # Example: "a held item: Book_3d15d052 with something if agent rotates Right 90 degrees" # noqa: E800
        if "a held item" in msg and "if agent rotates" in msg:
            return "The held item will collide with something if the robot turns that direction."
        # Example: "No valid positions to place object found" # noqa: E800
        if "No valid positions to place" in msg:
            return "The receptacle is too full or too small to contain the held item."
        # Example: "This target object is NOT a receptacle!" # noqa: E800
        if "NOT a receptacle" in msg:
            return "Object is not a receptacle the robot can place items in."
        # Example: "Target must be OFF to open!" # noqa: E800
        if "OFF to open!" in msg:
            return "Object must be turned off before it can be opened."
        # Example: "cracked_egg_5(Clone) is not a valid Object Type to be placed in StoveBurner_58b674c4" # noqa: E800
        if "not a valid Object Type to be placed" in msg:
            return "Held object cannot be placed there."
        # Example: "No target found" # noqa: E800
        if "No target found" in msg:
            return "No reachable object at that location."
        # Example: "Knife|-01.70|+01.71|+04.01 is not interactable and (perhaps it is occluded by something)." # noqa: E800
        if "it is occluded by something" in msg:
            return "An object is blocking you from interacting with the selected object."
        # "Could not find a target object at the specified location" # noqa: E800
        if "Could not find a target object" in msg:
            return "No valid object at that location."
        # "another object's collision is blocking held object from being placed" # noqa: E800
        if "another object's collision is blocking" in msg:
            return "The target area is too cluttered or the held object is already colliding with something."
        # "CounterTop|+00.69|+00.95|-02.48 is too far away to be interacted with" # noqa: E800
        if "too far away to" in msg:
            return "That object is too far away to interact with."
        # "Your partner is too far away for a handoff." # noqa: E800
        if "too far away for" in msg:
            return "Your partner is too far away for a handoff."
        # "Place: ObjectInteraction only supported when holding an object" # noqa: E800
        if "only supported when holding" in msg:
            return "You are not holding an object."
        # "Picking up object would cause it to collide and clip into something!" # noqa: E800
        if "would cause it to collide and" in msg:
            return "Cannot grab object from here without colliding with something."
        # "You cannot slice something while your partner is holding it." # noqa: E800
        if "cannot slice something while" in msg:
            return msg

        # If msg couldn't be handled, don't create a readable system message
        return None

    def get_hotspots(
        self,
        agent_id,
        hotspot_pixel_width=None,
        action_str=None,
        object_id=None,
        camera_id=None,
        return_full_seg_mask=False,
    ):
        """
        Return a segmentation mask highlighting object(s) in an egocentric image
        :param agent_id: the agent whose image needs to be highlighted; 0 for Commander and 1 for Driver/ Follower
        :param hotspot_pixel_width: Minimum hotspot size
        :param action_str: Highlight objects on which this action can be performed
        :param object_id: Specify object to be highlighted using object ID
        :param camera_id: Generate segmentation mask for a disembodied camera with this ID instead of for an agent
        :param return_full_seg_mask: additional flag to highlight a single object specified by object_id
        """
        assert not return_full_seg_mask or object_id is not None
        assert (action_str is None or object_id is None) and not (action_str is not None and object_id is not None)
        assert agent_id is None or camera_id is None
        if hotspot_pixel_width is None:
            hotspot_pixel_width = self.hotspot_pixel_width
        if agent_id is not None:
            sim_agent_id = agent_id if self.commander_embodied else 0
            le = (
                self.controller.last_event.events[sim_agent_id]
                if self.commander_embodied
                else self.controller.last_event
            )
            # Take a no-op step to render the object segmentation frame for hotspots.
            if le.instance_segmentation_frame is None:
                ac = dict(action="Pass", agentId=sim_agent_id, renderObjectImage=True)
                if debug_print_all_sim_steps:
                    logger.info("step %s", ac)
                self.controller.step(ac)
            if self.commander_embodied:
                le = self.controller.last_event.events[sim_agent_id]
                instance_segs = np.array(le.instance_segmentation_frame)
            elif agent_id == 0:  # commander camera
                le = self.controller.last_event
                instance_segs = np.array(le.third_party_instance_segmentation_frames[0])
            else:  # driver camera
                le = self.controller.last_event
                instance_segs = np.array(le.instance_segmentation_frame)
            color_to_object_id = le.color_to_object_id
            object_id_to_color = le.object_id_to_color
        else:
            le = self.controller.last_event.events[0] if self.commander_embodied else self.controller.last_event
            if le.instance_segmentation_frame is None:
                ac = dict(action="Pass", agentId=0, renderObjectImage=True)
                if debug_print_all_sim_steps:
                    logger.info("step %s", ac)
                self.controller.step(ac)
                le = self.controller.last_event.events[0] if self.commander_embodied else self.controller.last_event
            instance_segs = np.array(le.third_party_instance_segmentation_frames[camera_id])
            color_to_object_id = le.color_to_object_id
            object_id_to_color = le.object_id_to_color

        if return_full_seg_mask:
            mask = np.zeros_like(instance_segs, dtype=np.uint8)
            if object_id in object_id_to_color:
                color = object_id_to_color[object_id]
                mask = cv2.inRange(instance_segs, color, color)
                mask = np.array(mask) / 255
                mask = mask.astype(int)
            return {"mask": mask}
        else:
            hotspots = list()
            for x in range(0, self.web_window_size, hotspot_pixel_width):
                for y in range(0, self.web_window_size, hotspot_pixel_width):
                    instance_color_id = tuple(
                        instance_segs[y + hotspot_pixel_width // 2, x + hotspot_pixel_width // 2]
                    )  # coordinate system is y x
                    is_hotspot = False
                    if instance_color_id in color_to_object_id:  # anecdotally, some colors are missing from this map.
                        oid = color_to_object_id[instance_color_id]
                        obj = le.get_object(oid)
                        if action_str is not None:  # search by action str
                            affordance_lists = self.action_to_affordances[action_str]
                            if obj is not None and oid in le.instance_detections2D and obj["visible"]:
                                if np.any(
                                    [
                                        np.all([obj[prop] == affordances[prop] for prop in affordances])
                                        for affordances in affordance_lists
                                    ]
                                ):
                                    is_hotspot = True
                            elif (
                                self.commander_embodied
                                and action_str == "Place"
                                and oid[: len("agent_")] == "agent_"
                                and self.__agent_dist_to_agent(agent_id, (agent_id + 1) % 2) <= self.visibility_distance
                            ):
                                is_hotspot = True  # handoff to partner agent
                        elif object_id is not None:  # search by objectId
                            if obj is not None and obj["objectId"] == object_id:
                                is_hotspot = True
                    if is_hotspot:
                        hotspots.append([float(x) / self.web_window_size, float(y) / self.web_window_size])
            return {"hotspot_width": float(hotspot_pixel_width) / self.web_window_size, "hotspots": hotspots}

    def reset(self):
        """
        Reset the simulator to the initial state of self.current_episode
        """
        self.__launch_simulator(world=self.world, world_type=self.world_type)
        super().reset()

    def info(self, include_scenes=False, include_objects=False):
        """
        Obtain information about the current task and episode
        """
        d = super().info(include_scenes=include_scenes, include_objects=include_objects)
        d.update({"world_type": self.world_type, "world": self.world, "agent_poses": self.__get_agent_poses()})
        return d

    def select_random_world(self, world_type=None):
        """
        Select a random AI2-THOR floor plan, constrained to the specified world_type if provided
        :param world_type: One of "Kitchen", "Bedroom", "Bathroom" and "Living room" or None; if None all rooms will
        be considered
        """
        if world_type is None:
            world_type = random.choice(["Kitchen", "Living room", "Bedroom", "Bathroom"])
        world_type, scene_names = self.__get_available_scene_names(world_type=world_type)
        return world_type, random.choice(scene_names)

    def get_latest_images(self):
        """
        Return current images
        :return: {
                    "ego": Egocentric frame of driver/ follower,
                    "allo": Egocentric frame fo commander,
                    "targetobject": Target object view seen by commander
                    "semantic": Mask used to highlight an object in commander's target object view
                  }
        """
        if self.controller is None:
            return {}

        # Allows animations by getting newest frame (water, fire, etc.)
        ac = dict(action="Pass", agentId=0)
        if debug_print_all_sim_steps:
            logger.info("step %s", ac)
        self.controller.step(ac)
        if self.commander_embodied:
            ac = dict(action="Pass", agentId=1)
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            self.controller.step(ac)

        if self.commander_embodied:
            return {
                "ego": self.controller.last_event.events[1].frame,
                "allo": self.controller.last_event.events[0].frame,
                "targetobject": self.controller.last_event.events[0].third_party_camera_frames[
                    self.object_target_camera_idx
                ],
                "semantic": self.controller.last_event.events[1].instance_segmentation_frame,
            }
        else:
            return {
                "ego": self.controller.last_event.frame,
                "allo": self.controller.last_event.third_party_camera_frames[0],
                "targetobject": self.controller.last_event.third_party_camera_frames[self.object_target_camera_idx],
                "semantic": self.controller.last_event.instance_segmentation_frame,
            }

    def go_to_pose(self, pose):
        """
        Teleport the agent to a desired pose
        :param pose: Desired target pose; instance of class Pose defined in dataset.py
        """
        if pose is None:
            return
        ac = dict(
            action="TeleportFull",
            agentId=1 if self.commander_embodied else 0,
            x=pose.x,
            y=pose.y,
            z=pose.z,
            rotation=dict(x=0.0, y=pose.y_rot, z=0.0),
            horizon=pose.x_rot,
        )
        if debug_print_all_sim_steps:
            logger.info("step %s", ac)
        self.controller.step(ac)

    def get_current_pose(self, agent_id=None):
        """
        Return agent's current pose in the form of a Pose object
        :param agent_id: 0 for Commander and 1 for Driver/ Follower
        """
        event = self.controller.last_event.events[agent_id] if self.commander_embodied else self.controller.last_event
        position = event.metadata["agent"]["position"]
        rotation = event.metadata["agent"]["rotation"]
        horizon = event.metadata["agent"]["cameraHorizon"]

        return Pose.from_array([position["z"], -position["x"], position["y"], 0, horizon, -rotation["y"]])

    def get_available_scenes(self):
        """
        Load list of AI2-THOR floor plans
        """
        with importlib.resources.open_text(config_directory, "metadata_ai2thor.json") as f:
            data = json.load(f)
        return data

    def get_available_objects(self):
        """
        Load list of AI2-THOR objects
        """
        data = None
        with importlib.resources.open_text(config_directory, "metadata_google_scanned_objects.json") as f:
            data = json.load(f)

        return data

    def __get_agent_click_normalized_position(self, agent_id=None, agent_metadata=None):
        """
        Convert agent position to a visual coordinate on topdown map for TEACh data collection
        :param agent_id: 0 for Commander and 1 for Driver/ Follower
        :param agent_metadata: Pass agent metadata from a specific simulator event if desired
        """
        if agent_id is None:
            e = self.controller.last_event
        else:
            e = self.controller.last_event.events[agent_id]
        if agent_metadata is None:
            agent_metadata = e.metadata["agent"]

        ax = agent_metadata["position"]["x"]
        az = agent_metadata["position"]["z"]
        return self.__get_click_normalized_position_from_xz(ax, az)

    def __get_click_normalized_position_from_xz(self, x, z):
        """
        Convert AI2-THOR x, z coordinate to a visual coordinate on topdown map for
        TEACh data collection
        :param x: x coordinate on AI2-THOR floor plan
        :param z: z coordinate on AI2-THOR floor plan
        """
        norm_x, norm_z = (np.array((x, z)) - self.topdown_lower_left_xz) / (2 * self.topdown_cam_orth_size)
        click_x, click_y = norm_x, (1 - norm_z)  # z is flipped from top-to-bottom y of UI, so 1 - y = z
        return click_x, click_y

    def __get_agent_click_rotation(self, agent_id=None, agent_metadata=None):
        """
        Convert agent rotation to a visual view cone on topdown map for TEACh data collection
        :param agent_id: 0 for Commander and 1 for Driver/ Follower
        :param agent_metadata: Pass agent metadata from a specific simulator event if desired
        """
        if agent_metadata is None:
            if agent_id is None:
                e = self.controller.last_event
            else:
                e = self.controller.last_event.events[agent_id]
            agent_metadata = e.metadata["agent"]

        agent_y_rot = agent_metadata["rotation"]["y"]
        s_rot = self.__get_xz_rot_from_y(agent_y_rot)
        return s_rot[0], -s_rot[1]  # y flips z in AI2THOR

    def __get_xz_rot_from_y(self, y):
        """
        Given degrees y in [0, 359], return the closest cardinal direction as a tuple (x_dir, z_dir) in {-1, 0, 1}^2
        :param y: Input angle in [0, 359]
        """
        dir_degrees = [270, 180, 90, 0]
        closest_degree = dir_degrees[min(range(len(dir_degrees)), key=lambda i: abs(dir_degrees[i] - y))]
        if closest_degree == 270:  # facing x negative, z neutral
            s_rot = (-1, 0)
        elif closest_degree == 180:  # facing x neutral, z negative
            s_rot = (0, -1)
        elif closest_degree == 90:  # facing x positive, z neutral
            s_rot = (1, 0)
        else:  # facing x neutral, z positive
            s_rot = (0, 1)
        return s_rot

    def __get_y_rot_from_xz(self, x, z):
        """
        Given (x, z) norm rotation (e.g., (0, 1)), return the closest degrees in [270, 180, 90, 0] matching.
        """
        s_rot_to_dir_degree = {(-1, 0): 270, (0, -1): 180, (1, 0): 90, (0, 1): 0}
        return s_rot_to_dir_degree[(x, z)]

    def __get_available_scene_names(self, world_type=None):
        """
        Return available AI2-THOR floor plans, restricting to world_type if provided
        :param world_type: One of "Kitchen", "Bedroom", "Bathroom", "Living room" or None
        """
        if world_type is None:
            world_type = self.world_type

        data = self.get_available_scenes()
        scene_names = []
        if data is not None:
            for group in data["supported_worlds"]:
                if group["world_type"] == world_type:
                    for world in group["worlds"]:
                        scene_names.append(world["name"])
                    break  # done

        return world_type, scene_names

    def __get_world_type(self, world):
        """
        Given an AI2-THOR floor plan name, return the world type
        :param world: input floor plan name
        :return: One of "Kitchen", "Bedroom", "Bathroom", "Living room"
        """
        world_type = None
        try:
            number = int(world.split("_")[0][9:])  # Example: floor plan27_physics
            room_lo_hi = [("Kitchen", 1, 31), ("Living room", 201, 231), ("Bedroom", 301, 331), ("Bathroom", 401, 431)]
            for current_world_type, low, high in room_lo_hi:
                if number >= low and number <= high:
                    world_type = current_world_type
                    break
        except Exception as e:
            self.logger.warning(str(e))

        return world_type

    def __initialize_oracle_view(self):
        """
        Set up third party camera for TEACh data collection
        """
        pose_robot = self.get_current_pose(agent_id=0)
        ac = dict(
            action="AddThirdPartyCamera",
            rotation=dict(x=30, y=-pose_robot.z_rot, z=0),  # Look down at 30 degrees
            position=dict(x=-pose_robot.y, y=pose_robot.z + 1, z=pose_robot.x),
            fieldOfView=90,
        )
        if debug_print_all_sim_steps:
            logger.info("step %s", ac)
        self.controller.step(ac)

    def shutdown_simulator(self):
        """
        Stop AI2-THOR Unity process; call when done using simulator
        """
        if self.controller is not None:
            self.controller.stop()
        self.controller = None

    def __launch_simulator(self, world=None, world_type=None):
        """
        Initialize simulator for a new episode
        """
        time_start = time.time()
        need_new_map = False
        if self.world is None and world is None:  # no presets and no args, so choose randomly.
            if world_type in ["Kitchen", "Living room", "Bedroom", "Bathroom"]:
                self.world_type, self.world = self.__get_available_scene_names(world_type=world_type)
            else:
                self.world_type, self.world = self.select_random_world()
            need_new_map = True
        elif world is not None:  # set world/type by args.
            if self.world != world:
                need_new_map = True
            self.world = world
            self.world_type = self.__get_world_type(world)

        if self.controller is not None:
            self.controller.stop()

        init_params = dict(
            base_dir=self.controller_base_dir,
            local_executable_path=self.controller_local_executable_path,
            scene=self.world,
            gridSize=self.grid_size,
            snapToGrid=True,
            visibilityDistance=self.visibility_distance,
            width=self.web_window_size,
            height=self.web_window_size,
            agentCount=2 if self.commander_embodied else 1,
            commit_id=COMMIT_ID,
        )

        logger.info("In SimulatorTHOR.__launch_simulator, creating ai2thor controller (unity process)")
        time_start_controller = time.time()
        if debug_print_all_sim_steps:
            logger.info("init %s", init_params)
        self.controller = TEAChController(**init_params)
        time_end_controller = time.time()
        self.logger.info("Time to create controller: %s sec" % (time_end_controller - time_start_controller))

        # Tilt agents down.
        ac = dict(action="LookDown", agentId=0, degrees=30)
        if debug_print_all_sim_steps:
            logger.info("step %s", ac)
        self.controller.step(ac)
        if self.commander_embodied:
            ac = dict(action="LookDown", agentId=1, degrees=30)
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            self.controller.step(ac)

        # Get topdown map camera details used to turn MapGoal (x, y) clicks into (x, z) sim coords.
        if need_new_map:
            ac = {"action": "ToggleMapView"}
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            self.controller.step(ac)
            topdown_cam_position = self.controller.last_event.metadata["cameraPosition"]
            self.topdown_cam_orth_size = self.controller.last_event.metadata["cameraOrthSize"]
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            self.controller.step(ac)
            self.topdown_lower_left_xz = (
                np.array((topdown_cam_position["x"], topdown_cam_position["z"])) - self.topdown_cam_orth_size
            )

            self.navigation_graph = None  # Clear cached nav graph if any

        self.is_ready = True
        # Initialize 3rd party camera for disembodied commander.
        if not self.commander_embodied:
            self.__initialize_oracle_view()
            self.object_target_camera_idx = 1
        else:
            self.object_target_camera_idx = 0
        # Initialize a 3rd party camera for object targetting (idx 0 if embodied commander, 1 else).
        self.controller.step(
            "AddThirdPartyCamera", rotation=dict(x=0, y=0, z=90), position=dict(x=-1.0, z=-2.0, y=1.0), fieldOfView=90
        )

        # Get floor oid.
        self.floor_oid = self.__get_nearest_object_matching_search_str("Floor")["objectId"]

        time_end = time.time()
        self.logger.info("Time to launch simulator: %s sec" % (time_end - time_start))
        self.logger.debug("Launched world: %s; commander embodied: %s" % (world, str(self.commander_embodied)))

    def randomize_agent_positions(self):
        """
        Randomize the positions of the agents in the current scene.
        """
        ac = dict(action="GetReachablePositions")
        if debug_print_all_sim_steps:
            logger.info("step %s", ac)
        event = self.controller.step(ac)
        all_points = event.metadata["actionReturn"]
        target_points = None
        d = None
        while d is None or d <= self.grid_size * 2:
            target_points = list(np.random.choice(all_points, size=2, replace=False))
            d = np.linalg.norm([target_points[0][c] - target_points[1][c] for c in ["x", "z"]])
        locs = [
            {
                "position": {
                    "x": target_points[idx]["x"],
                    "y": event.metadata["agent"]["position"]["y"],
                    "z": target_points[idx]["z"],
                },
                "rotation": {
                    "x": event.metadata["agent"]["rotation"]["x"],
                    "y": self.__get_y_rot_from_xz(*[(-1, 0), (0, -1), (1, 0), (0, 1)][np.random.randint(0, 4)]),
                    "z": event.metadata["agent"]["rotation"]["z"],
                },
                "cameraHorizon": event.metadata["agent"]["cameraHorizon"],
            }
            for idx in range(2)
        ]
        return self.set_agent_poses(locs)

    def randomize_scene_objects_locations(
        self, n_placement_attempts=5, min_duplicates=1, max_duplicates=4, duplicate_overrides=None
    ):
        """
        Randomize the objects in the current scene.
        Resets the scene, so object states will not survive this call.
        https://ai2thor.allenai.org/ithor/documentation/actions/initialization/#object-position-randomization
        :param n_placement_attempts: how many times to try to place each object (larger just takes longer)
        :param min_duplicates: minimum number of each object type in the original scene to keep
        :param max_duplicates: maximum number of each object type in the original scene to duplicate
        :param duplicate_overrides: dict; override numDuplicatesOfType with key value pairs here for keys in override
        """
        otypes = set()
        for obj in self.get_objects(self.controller.last_event):
            otypes.add(obj["objectType"])
        duplicates = [
            {
                "objectType": ot,
                "count": np.random.randint(
                    duplicate_overrides[ot]
                    if duplicate_overrides is not None and ot in duplicate_overrides
                    else min_duplicates,
                    max(max_duplicates, duplicate_overrides[ot] + 1)
                    if duplicate_overrides is not None and ot in duplicate_overrides
                    else max_duplicates,
                ),
            }
            for ot in otypes
        ]
        ac = dict(
            action="InitialRandomSpawn",
            randomSeed=np.random.randint(0, 1000),
            numPlacementAttempts=n_placement_attempts,
            placeStationary=True,
            numDuplicatesOfType=duplicates,
        )
        if debug_print_all_sim_steps:
            logger.info("step %s", ac)
        event = self.controller.step(ac)

        # Make objects unbreakable to prevent shattering plates, etc on Place that uses PutObjectAtPoint.
        breakable_ots = list(set([obj["objectType"] for obj in self.get_objects() if obj["breakable"]]))
        for ot in breakable_ots:
            ac = dict(action="MakeObjectsOfTypeUnbreakable", objectType=ot)
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            self.controller.step(ac)

        return event.metadata["lastActionSuccess"], event.metadata["errorMessage"]

    def randomize_scene_objects_states(self):
        """
        Randomize some object states for objects in the current scene.
        """
        otypes_to_states = {}
        randomize_attrs = {
            "toggleable": ["isToggled", "ToggleObjectOn", "ToggleObjectOff"],
            "canFillWithLiquid": ["isFilledWithLiquid", "FillObjectWithLiquid", "EmptyLiquidFromObject"],
            "dirtyable": ["isDirty", "DirtyObject", "CleanObject"],
            "canBeUsedUp": ["isUsedUp", "UseUpObject", None],
        }
        for obj in self.get_objects(self.controller.last_event):
            ot = obj["objectType"]
            if ot not in otypes_to_states:
                otypes_to_states[ot] = {attr for attr in randomize_attrs if obj[attr]}
        success = True
        msgs = []
        for obj in self.get_objects(self.controller.last_event):
            for attr in otypes_to_states[obj["objectType"]]:
                state = np.random.random() < 0.5
                if (state and randomize_attrs[attr][1] is not None) or (
                    not state and randomize_attrs[attr][2] is not None
                ):
                    if obj[randomize_attrs[attr][0]] != state:
                        action = dict(
                            action=randomize_attrs[attr][1 if state else 2], objectId=obj["objectId"], forceAction=True
                        )
                        if action["action"] == "FillObjectWithLiquid":
                            action["fillLiquid"] = "water"
                            # if obj['objectType'] == 'Mug':
                            #     continue
                        if action["action"] == "ToggleObjectOn" and obj["breakable"] and obj["isBroken"]:
                            continue  # e.g., if a laptop is broken, it cannot be turned on
                        if debug_print_all_sim_steps:
                            logger.info("step %s", action)
                        event = self.controller.step(action)
                        if not event.metadata["lastActionSuccess"]:
                            success = False
                            msgs.append([action, event.metadata["errorMessage"]])
        return success, "\n".join(["%s: %s" % (msgs[idx][0], msgs[idx][1]) for idx in range(len(msgs))])

    def get_scene_object_locs_and_states(self):
        """
        Return all the object metadata and agent position data from AI2-THOR.
        """
        if self.commander_embodied:
            a = [
                self.controller.last_event.events[0].metadata["agent"],
                self.controller.last_event.events[1].metadata["agent"],
            ]
        else:
            a = [
                self.controller.last_event.metadata["thirdPartyCameras"][0],
                self.controller.last_event.metadata["agent"],
            ]
        return {"objects": self.get_objects(self.controller.last_event), "agents": a}

    def restore_scene_object_locs_and_states(self, objs):
        """
        Restore all the object positions, rotations, and states saved.
        :param objs: Object positions and states
        """
        # Restore instances and locations.
        success = True
        msgs = []
        object_poses = []
        scene_objs = self.get_objects()
        for obj in objs:
            if obj["pickupable"] or obj["moveable"]:
                obj_name = obj["name"][: obj["name"].index("(") if "(" in obj["name"] else len(obj["name"])]
                if np.any(
                    [
                        obj_name
                        == s_obj["name"][: s_obj["name"].index("(") if "(" in s_obj["name"] else len(s_obj["name"])]
                        for s_obj in scene_objs
                    ]
                ):
                    object_poses.append(
                        {"objectName": obj_name, "rotation": dict(obj["rotation"]), "position": dict(obj["position"])}
                    )
        action = dict(
            action="SetObjectPoses",
            # cut off "(Copy)..." from object name
            objectPoses=object_poses,
            placeStationary=True,
        )
        if debug_print_all_sim_steps:
            logger.info("step %s", action)
        event = self.controller.step(action)
        if not event.metadata["lastActionSuccess"]:
            success = False
            msgs.append([action["action"], event.metadata["errorMessage"]])
        # Restore object states.
        restore_attrs = {
            "toggleable": ["isToggled", "ToggleObjectOn", "ToggleObjectOff"],
            "canFillWithLiquid": ["isFilledWithLiquid", "FillObjectWithLiquid", "EmptyLiquidFromObject"],
            "dirtyable": ["isDirty", "DirtyObject", "CleanObject"],
            "openable": ["isOpen", "OpenObject", "CloseObject"],
            "canBeUsedUp": ["isUsedUp", "UseUpObject", None],
            "sliceable": ["isSliced", "SliceObject", None],
            "cookable": ["isCooked", "CookObject", None],
            "breakable": ["isBroken", "BreakObject", None],
        }
        scene_objs = self.get_objects(self.controller.last_event)
        for obj in objs:
            for attr in restore_attrs:
                attr_state, attr_on, attr_off = restore_attrs[attr]
                if obj[attr]:
                    scene_obj = self.__get_object_by_id(scene_objs, obj["objectId"])
                    if not scene_obj:
                        scene_obj = self.__get_object_by_position(scene_objs, obj["position"])
                    if scene_obj["objectType"] != obj["objectType"]:
                        success = False
                        msgs.append(["restore states", "could not find scene obj for %s" % obj["objectId"]])
                        continue
                    if obj[attr_state] != scene_obj[attr_state]:
                        action = dict(
                            action=attr_on if obj[attr_state] else attr_off,
                            objectId=scene_obj["objectId"],
                            forceAction=True,
                        )
                        if action["action"] is None:
                            success = False
                            msgs.append(
                                [
                                    "restore states",
                                    "unable to take action to remedy object "
                                    + "%s wants state %s=%s while scene obj has state %s"
                                    % (obj["objectId"], attr_state, str(obj[attr_state]), str(scene_obj[attr_state])),
                                ]
                            )
                            continue
                        if action["action"] == "FillObjectWithLiquid":
                            action["fillLiquid"] = "water"
                        if debug_print_all_sim_steps:
                            logger.info("step %s", action)
                        event = self.controller.step(action)
                        if not event.metadata["lastActionSuccess"]:
                            success = False
                            msgs.append([action, event.metadata["errorMessage"]])

        return success, "\n".join(["%s: %s" % (msgs[idx][0], msgs[idx][1]) for idx in range(len(msgs))])

    def restore_initial_state(self):
        """
        Reset the simulator to initial state of current episode
        """
        _, succ = self.load_scene_state(init_state=self.current_episode.initial_state)
        return succ

    def load_scene_state(self, fn=None, init_state=None):
        """
        Reset start time and init state.
        :param fn: Filename to load initial state from
        :param init_state: Valid initial state to initialize simulator with; must be an instance of class
        Initialization in dataset.py
        """
        loaded_fn, succ = super().load_scene_state(fn=fn, init_state=init_state)
        # Make objects unbreakable to prevent shattering plates, etc on Place that uses PutObjectAtPoint.
        breakable_ots = list(set([obj["objectType"] for obj in self.get_objects() if obj["breakable"]]))
        for ot in breakable_ots:
            ac = dict(action="MakeObjectsOfTypeUnbreakable", objectType=ot)
            if debug_print_all_sim_steps:
                logger.info("step %s", ac)
            self.controller.step(ac)
        return loaded_fn, succ

    def set_init_state(self):
        """
        Set the initial state of the episode to current state.
        """
        self.current_episode.initial_state = self.get_current_state()
        super().set_init_state()

    def get_current_state(self):
        """
        Return current state in the form of an instance of class Initialization defined in dataset.py
        """
        self.__add_obj_classes_for_objs()   # Add object classes for any newly created objects (eg: slices)
        self.__check_per_step_custom_properties()   # Confirm whether any updates to custom properties need to be made
                                                    # from last time step
        state = self.get_scene_object_locs_and_states()
        return Initialization(
            time_start=self.start_time,
            agents=state["agents"],
            objects=state["objects"],
            custom_object_metadata=self.__custom_object_metadata,
        )

    def __get_object_by_position(self, m, pos, obj_type=None, ignore_object_ids=None):
        """
        Get the object closet to the given position.
        :param m: output of get_objects()
        :param pos: object x, y, z dict pose
        :param obj_type: nearest object of a particular type, or None for any
        """
        o = None
        d = None
        for obj in [_obj for _obj in m if obj_type is None or _obj["objectType"] == obj_type]:
            if ignore_object_ids is not None and obj["objectId"] in ignore_object_ids:
                continue
            obj_pos = obj["position"]
            _d = np.linalg.norm([pos["x"] - obj_pos["x"], pos["y"] - obj_pos["y"], pos["z"] - obj_pos["z"]])
            if d is None or _d < d:
                d = _d
                o = obj
        return o

    def __get_object_by_id(self, m, obj_id):
        """
        Get the object matching the id.
        :param m: output of get_objects()
        :param obj_id: id to match
        """
        for obj in m:
            if obj["objectId"] == obj_id:
                return obj
        return False

    def set_agent_poses(self, locs):
        """
        Set agents to specified poses
        :param locs: Desired agent poses
        """
        success = True
        msgs = []
        if self.commander_embodied:
            for idx in range(2):
                action = dict(
                    action="Teleport",
                    agentId=idx,
                    x=locs[idx]["position"]["x"],
                    y=locs[idx]["position"]["y"],
                    z=locs[idx]["position"]["z"],
                    rotation=dict(
                        x=locs[idx]["rotation"]["x"], y=locs[idx]["rotation"]["y"], z=locs[idx]["rotation"]["z"]
                    ),
                    horizon=locs[idx]["cameraHorizon"],
                )
                if debug_print_all_sim_steps:
                    logger.info("step %s", action)
                event = self.controller.step(action)
                if not event.metadata["lastActionSuccess"]:
                    success = False
                    msgs.append([action["action"], event.metadata["errorMessage"]])
        else:
            action = dict(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=0,
                rotation=dict(x=locs[0]["rotation"]["x"], y=locs[0]["rotation"]["y"], z=locs[0]["rotation"]["z"]),
                position=dict(x=locs[0]["position"]["x"], y=locs[0]["position"]["y"], z=locs[0]["position"]["z"]),
            )
            if debug_print_all_sim_steps:
                logger.info("step %s", action)
            event = self.controller.step(action)
            if not event.metadata["lastActionSuccess"]:
                success = False
                msgs.append([action["action"], event.metadata["errorMessage"]])
            action = dict(
                action="Teleport",
                x=locs[1]["position"]["x"],
                y=locs[1]["position"]["y"],
                z=locs[1]["position"]["z"],
                rotation=dict(x=locs[1]["rotation"]["x"], y=locs[1]["rotation"]["y"], z=locs[1]["rotation"]["z"]),
                horizon=locs[1]["cameraHorizon"],
            )
            if debug_print_all_sim_steps:
                logger.info("step %s", action)
            event = self.controller.step(action)
            if not event.metadata["lastActionSuccess"]:
                success = False
                msgs.append([action["action"], event.metadata["errorMessage"]])
        return success, "\n".join(["%s: %s" % (msgs[idx][0], msgs[idx][1]) for idx in range(len(msgs))])
