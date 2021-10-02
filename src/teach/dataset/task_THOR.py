# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import copy
import importlib.resources
import json
import logging
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

from teach.dataset.episode import Episode
from teach.dataset.task import Task
from teach.logger import create_logger

logger = create_logger(__name__, logging.WARNING)


class Task_THOR(Task):
    def __init__(
        self,
        task_id,
        task_name,
        task_nparams,
        task_params,
        task_anchor_object,
        desc,
        components,
        relations,
        comments="",
        episodes=None,
    ):
        subgoals = dict()
        subgoals["components"] = components
        subgoals["relations"] = relations
        super().__init__(task_id, task_name, task_nparams, task_params, subgoals, comments, episodes)
        self.task_id = task_id
        self.task_name = task_name
        self.task_nparams = task_nparams
        self.task_params = task_params
        self.task_anchor_object = task_anchor_object
        self.desc = desc
        self.components = components
        self.relations = relations
        self.comments = comments
        self.episodes = [] if episodes is None else episodes

    @staticmethod
    def component_to_dict(component):
        if "task_name" not in component:
            return component
        else:
            component_dict = copy.deepcopy(component)
            component_dict["task"] = component["task"].to_dict()
            return component_dict

    def to_dict(self):
        _dict = OrderedDict()
        _dict["task_id"] = self.task_id
        _dict["task_name"] = self.task_name
        _dict["task_params"] = self.task_params
        _dict["task_nparams"] = self.task_nparams
        _dict["task_anchor_object"] = self.task_anchor_object
        _dict["desc"] = self.desc
        _dict["components"] = dict()
        for component_key, component in self.components.items():
            component_dict = self.component_to_dict(component)
            _dict["components"][component_key] = component_dict
        _dict["relations"] = self.relations
        _dict["comments"] = self.comments
        _dict["episodes"] = [x.to_dict() for x in self.episodes]
        return _dict

    @classmethod
    def from_dict(cls, task_dict, definitions, process_init_state=True) -> "Task_THOR":
        episodes = [
            Episode.from_dict(episode_dict, definitions, process_init_state)
            for episode_dict in task_dict.get("episodes")
        ]
        return cls(
            task_id=task_dict["task_id"],
            task_name=task_dict["task_name"],
            task_nparams=task_dict["task_nparams"],
            task_params=task_dict["task_params"],
            task_anchor_object=task_dict["task_anchor_object"],
            desc=task_dict["desc"],
            components=task_dict["components"],
            relations=task_dict["relations"],
            comments=task_dict.get("comments"),
            episodes=episodes,
        )

    @classmethod
    def from_v1_dict(cls, task_dict, definitions, process_init_state=True) -> "Task_THOR":
        episodes = [
            Episode.from_dict(episode_dict, definitions, process_init_state)
            for episode_dict in task_dict.get("episodes")
        ]
        return cls(
            task_id=task_dict["task_id"],
            task_name=task_dict["task_name"],
            task_nparams=task_dict["task_nparams"],
            task_params=task_dict["task_params"],
            task_anchor_object=None,
            desc="Complete the following tasks.",
            components=dict(enumerate(task_dict["subgoals"])),
            relations=[],
            comments=task_dict.get("comments"),
            episodes=episodes,
        )

    @staticmethod
    def load_tasks(resource_package):
        """
        Given a directory with
        """
        tasks = list()
        task_id_to_task_dict = dict()
        task_name_to_task_dict = dict()
        task_dependencies = dict()
        resolved_task_names = set()
        required_keys = [
            "task_id",
            "task_name",
            "task_nparams",
            "task_anchor_object",
            "desc",
            "components",
            "relations",
        ]
        for task_file in importlib.resources.contents(resource_package):
            if not importlib.resources.is_resource(resource_package, task_file):
                continue
            if not task_file.endswith(".json"):
                continue

            logger.info("Processing file %s" % task_file)
            with importlib.resources.open_text(resource_package, task_file) as file_handle:
                task_definition = json.load(file_handle)
                if type(task_definition) != dict:
                    raise RuntimeError(
                        "Badly formatted task file: "
                        + str(task_file)
                        + ". Each task file must be a json dictionary with keys: "
                        + str(required_keys)
                    )
                for key in required_keys:
                    if key not in task_definition.keys():
                        raise RuntimeError("Badly formatted task file. Missing key:" + str(key))
                task = Task_THOR(
                    task_id=task_definition["task_id"],
                    task_name=task_definition["task_name"],
                    task_nparams=task_definition["task_nparams"],
                    task_params=None,
                    task_anchor_object=task_definition["task_anchor_object"],
                    desc=task_definition["desc"],
                    components=task_definition["components"],
                    relations=task_definition["relations"],
                    comments="",
                    episodes=None,
                )
                tasks.append(task)
                if task.task_id in task_id_to_task_dict.keys():
                    raise RuntimeError(
                        "Duplicate task_id " + str(task.task_id) + " with one occurrence in " + str(task_file)
                    )
                if task.task_name in task_name_to_task_dict.keys():
                    raise RuntimeError(
                        "Duplicate task_name " + str(task.task_name) + " with one occurrence in " + str(task_file)
                    )
                task_id_to_task_dict[task.task_id] = task
                task_name_to_task_dict[task.task_name] = task

                task_dependencies[task.task_name] = list()
                resolved = True
                for component_name, component_dict in task.components.items():
                    if "task_name" in component_dict:
                        resolved = False
                        task_dependencies[task.task_name].append(component_dict["task_name"])
                    else:
                        atomic_component_keys = {
                            "determiner",
                            "primary_condition",
                            "instance_shareable",
                            "conditions",
                            "condition_failure_descs",
                        }
                        if len(atomic_component_keys.difference(set(component_dict.keys()))) > 0:
                            raise RuntimeError(
                                "Improperly defined component "
                                + str(component_name)
                                + " in task "
                                + str(task.task_name)
                                + ". Must contain keys: "
                                + str(atomic_component_keys)
                            )
                if resolved:
                    resolved_task_names.add(task.task_name)

        logger.info("Loaded task names: %s", task_name_to_task_dict.keys())

        # Resolve task dependencies
        unresolved_tasks = set()
        unresolvable_tasks = set()
        for task in tasks:
            resolved = True
            for component_name, component_dict in task.components.items():
                if "task_name" in component_dict:
                    if component_dict["task_name"] not in task_name_to_task_dict:
                        unresolvable_tasks.add((task.task_name, component_dict["task_name"]))
                        resolved = False
                        break

                    if component_dict["task_name"] in resolved_task_names:
                        task.components[component_name]["task"] = copy.deepcopy(
                            task_name_to_task_dict[component_dict["task_name"]]
                        )
                        task.components[component_name]["task"].task_params = component_dict["task_params"]
                    else:
                        unresolved_tasks.add(task.task_name)
                        resolved = False
                        break
            if resolved:
                resolved_task_names.add(task.task_name)

        if len(unresolvable_tasks) > 0:
            error_msg = "Could not resolve the following tasks: " + "\n\t".join(
                [
                    'Subtask "' + str(dependency) + '" in task "' + str(task_name) + '"'
                    for (task_name, dependency) in unresolvable_tasks
                ]
            )
            raise RuntimeError(error_msg)

        while len(unresolved_tasks) > 0:
            logger.info("Still resolving tasks: %s", unresolved_tasks)
            for unresolved_task_name in unresolved_tasks:
                task = task_name_to_task_dict[unresolved_task_name]
                resolved = True
                for component_name, component_dict in task.components.items():
                    if "task_name" in component_dict:
                        if component_dict["task_name"] in resolved_task_names:
                            task.components[component_name]["task"] = copy.deepcopy(
                                task_name_to_task_dict[component_dict["task_name"]]
                            )
                            task.components[component_name]["task"].task_params = component_dict["task_params"]
                        else:
                            resolved = False
                            break
                if resolved:
                    resolved_task_names.add(task.task_name)
            unresolved_tasks = unresolved_tasks.difference(resolved_task_names)

        return tasks, task_id_to_task_dict, task_name_to_task_dict

    @staticmethod
    def __get_files_recursive(root_dir, file_list, extension=".json"):
        for path in Path(root_dir).iterdir():
            if path.is_dir():
                Task_THOR.__get_files_recursive(path, file_list)
            elif os.path.isfile(path) and path.suffix == extension:
                file_list.append(path.resolve())

    def __write_task_params_into_str(self, s):
        # Need to go through params in reverse order so that multiple digits get treated correctly
        for idx in range(len(self.task_params) - 1, -1, -1):
            s = s.replace("#%d" % idx, self.task_params[idx])
        return s

    def __write_task_params_into_list(self, task_params_list):
        for idx, elem in enumerate(task_params_list):
            if type(elem) == str:
                task_params_list[idx] = self.__write_task_params_into_str(elem)
            elif type(elem) == list:
                task_params_list[idx] = self.__write_task_params_into_list(elem)
            elif type(elem) == dict:
                task_params_list[idx] = self.__write_task_params_into_dict(elem)
            elif isinstance(elem, Task_THOR):
                elem.write_task_params()
        return task_params_list

    def __write_task_params_into_dict(self, d):
        keys_to_delete = list()
        dict_items = list(d.items())
        for key, value in dict_items:
            key_with_params = self.__write_task_params_into_str(key)
            if key_with_params != key:
                keys_to_delete.append(key)
            if type(value) == str:
                d[key_with_params] = self.__write_task_params_into_str(value)
                # if the value is a variable that just got filled, they key is not a determiner and the value is numeric
                if np.char.isnumeric(d[key_with_params]) and key_with_params not in ["determiner"]:
                    # then this is a variable indicating the value of a simulator property that needs to be int
                    d[key_with_params] = int(d[key_with_params])
            elif type(value) == list:
                d[key_with_params] = self.__write_task_params_into_list(value)
            elif type(value) == dict:
                d[key_with_params] = self.__write_task_params_into_dict(value)
            elif isinstance(value, Task_THOR):
                value.write_task_params()
                d[key_with_params] = value
        for key in keys_to_delete:
            del d[key]
        return d

    def write_task_params(self):
        try:
            assert len(self.task_params) == self.task_nparams
        except AssertionError as e:
            logger.error(
                f"Task {self.task_name} takes {self.task_nparams} params but supplied {len(self.task_params)}",
                exc_info=True,
            )
            raise e
        self.desc = self.__write_task_params_into_str(self.desc)
        if self.task_anchor_object is not None:
            self.task_anchor_object = self.__write_task_params_into_str(self.task_anchor_object)
        self.components = self.__write_task_params_into_dict(self.components)
        self.relations = self.__write_task_params_into_list(self.relations)

    def __get_object_by_id(self, m, obj_id):
        for obj in m:
            if obj["objectId"] == obj_id:
                return obj
        return False

    def get_parent_receptacles(self, obj, objects):
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

    def check_component_n_instances(
        self, all_objects_cur_state, component, num_instances, simulator=None, allow_state_change=False
    ):
        if component["instance_shareable"]:
            num_instances = 1
        component_success = False
        satisifed_objects = list()
        candidate_objects = list()  # Contains num_instances closest matches
        output = dict()
        num_instances_to_check = num_instances

        for obj in all_objects_cur_state:
            props_sat = self.obj_satisfies_props(obj, component["conditions"], all_objects_cur_state)
            props_sat_v = list(props_sat.values())

            if np.all(props_sat_v):
                satisifed_objects.append(obj)
                if len(satisifed_objects) >= num_instances:
                    component_success = True
                continue

            if not component_success:
                # Closet object must match objectType, then heuristically whichever matches most conditions.
                # Primary condition (e.g., objectType) must match.
                if (
                    props_sat[component["primary_condition"]]
                    or
                    # Or, if condition (e.g., objectType) is a slice, can match the base object condition var.
                    (
                        "Sliced" in component["conditions"][component["primary_condition"]]
                        and obj[component["primary_condition"]]
                        == component["conditions"][component["primary_condition"]].replace("Sliced", "")
                    )
                    or (
                        "Cracked" in component["conditions"][component["primary_condition"]]
                        and obj[component["primary_condition"]]
                        == component["conditions"][component["primary_condition"]].replace("Cracked", "")
                    )
                ):
                    # Either the primary condition is satisfied or can be with a state change
                    non_primary_prop_vals = [
                        value for (key, value) in props_sat.items() if key != component["primary_condition"]
                    ]
                    if (
                        np.all(non_primary_prop_vals)
                        and "Sliced" in component["conditions"][component["primary_condition"]]
                    ):
                        # We already checked that the primary condition would be satisfied with a state change
                        # And one object, say a potato can produce many slices
                        num_instances_to_check = 1
                        if allow_state_change:
                            satisifed_objects.append(obj)
                            if len(satisifed_objects) >= num_instances_to_check:
                                component_success = True
                            continue

                    if not component_success:
                        obj_dist = 0
                        if simulator is not None:
                            obj_dist = simulator.obj_dist_to_nearest_agent(obj)
                        obj_candidate_dict = {
                            "object": obj,
                            "props_sat": props_sat,
                            "num_props_sat": props_sat_v.count(True),
                            "distance_to_agent": obj_dist,
                        }
                        if len(candidate_objects) == 0:
                            candidate_objects.append(obj_candidate_dict)
                        else:
                            # Insert into sorted position
                            insert_position = None
                            for candidate_idx, cur_candidate in enumerate(candidate_objects):
                                if obj_candidate_dict["num_props_sat"] > cur_candidate[
                                    "num_props_sat"
                                ] or (  # Satisifes more
                                    obj_candidate_dict["num_props_sat"] == cur_candidate["num_props_sat"]
                                    and obj_candidate_dict["distance_to_agent"] < cur_candidate["distance_to_agent"]
                                ):  # Closer
                                    insert_position = candidate_idx
                                    break
                            if insert_position is not None:
                                # This is better than some existing candidate
                                candidate_objects.insert(insert_position, obj_candidate_dict)
                            else:
                                # Worse than all existing candidates
                                candidate_objects.append(obj_candidate_dict)

        num_unsatisfied_instances_needed = max(0, num_instances_to_check - len(satisifed_objects))

        output["success"] = component_success
        output["satisfied_objects"] = satisifed_objects
        output["candidate_objects"] = [candidate["object"] for candidate in candidate_objects]
        num_conditions_per_obj = len(component["conditions"].keys())
        output["goal_conditions_total"] = num_conditions_per_obj * num_instances
        output["steps"] = list()
        num_problem_objects = min(len(candidate_objects), num_unsatisfied_instances_needed)
        problem_objects = candidate_objects[:num_problem_objects]
        output["goal_conditions_satisfied"] = (
            num_conditions_per_obj * min(num_instances, len(satisifed_objects))
        ) + sum([candidate["num_props_sat"] for candidate in problem_objects])
        keys_to_problem_objects = dict()
        for candidate in problem_objects:
            for key in candidate["props_sat"]:
                if not candidate["props_sat"][key] and key in component["condition_failure_descs"]:
                    if key not in keys_to_problem_objects:
                        keys_to_problem_objects[key] = list()
                    keys_to_problem_objects[key].append(candidate)
        for key, desc in component["condition_failure_descs"].items():
            if key in keys_to_problem_objects:
                for candidate in keys_to_problem_objects[key]:
                    output["steps"].append(
                        {
                            "success": False,
                            "objectId": candidate["object"]["objectId"],
                            "objectType": candidate["object"]["objectType"],
                            "desc": desc,
                        }
                    )
            else:
                representative_obj = None
                if len(satisifed_objects) > 0:
                    representative_obj = satisifed_objects[0]
                elif len(candidate_objects) > 0:
                    representative_obj = candidate_objects[0]["object"]
                if representative_obj is not None:
                    output["steps"].append(
                        {
                            "success": True,
                            "objectId": representative_obj["objectId"],
                            "objectType": representative_obj["objectType"],
                            "desc": desc,
                        }
                    )
        output["problem_keys"] = dict()
        for candidate in problem_objects:
            output["problem_keys"][candidate["object"]["objectId"]] = list()
            for key in candidate["props_sat"]:
                if not candidate["props_sat"][key]:
                    output["problem_keys"][candidate["object"]["objectId"]].append(
                        {
                            "objectType": candidate["object"]["objectType"],
                            "determiner": component["determiner"],
                            "property_name": key,
                            "desired_property_value": component["conditions"][key],
                        }
                    )
        return output

    def check_episode_preconditions(self, simulator, all_objects_cur_state, num_instances_needed=1):
        """
        :param simulator: instance of Simulator_THOR
        :param all_objects_cur_state: List of dictionaries, each of which has key, value pairs corresponding to
        current properties of an object in the environment
        :param num_instances_needed: Only relevant for tasks with task_anchor_object != None - Sets the number of anchor
        objects to be created
        """
        self.write_task_params()

        for component in self.components.values():
            if component["determiner"] == "0":
                continue

            component_instances_needed = num_instances_needed
            if component["determiner"] == "all":
                component_instances_needed = 1
                allow_state_change = False
            else:
                if component["determiner"] != "a":
                    number_determiner = int(component["determiner"])
                    component_instances_needed *= number_determiner
                allow_state_change = True

            if "task_name" in component:
                component_success = component["task"].check_episode_preconditions(
                    simulator, all_objects_cur_state, component_instances_needed
                )
                if not component_success:
                    return False
            else:
                component_existence_dict = dict()
                component_existence_dict["primary_condition"] = component["primary_condition"]
                component_existence_dict["condition_failure_descs"] = component["condition_failure_descs"]
                component_existence_dict["instance_shareable"] = component["instance_shareable"]
                component_existence_dict["conditions"] = dict()
                component_existence_dict["conditions"][component["primary_condition"]] = component["conditions"][
                    component["primary_condition"]
                ]
                output = self.check_component_n_instances(
                    all_objects_cur_state,
                    component_existence_dict,
                    component_instances_needed,
                    simulator,
                    allow_state_change,
                )
                if not output["success"]:
                    return False
        return True

    @staticmethod
    def get_obj_by_id(obj_id, objects):
        for obj in objects:
            if obj["objectId"] == obj_id:
                return obj
        return None

    # Returns a list parallel to [props] of bools.
    def obj_satisfies_props(self, obj, props, all_objects):
        sats = {}
        for prop in props:
            # Property is not satisfied if the object doesn't even have it.
            if prop not in obj:
                sats[prop] = False
                continue

            if prop == "objectType":
                sats[prop] = self.check_object_type(obj, props[prop])
                continue
            elif prop == "simbotLastParentReceptacle" and props[prop] is not None:
                # need to match type / class rather than value
                value_obj = self.get_obj_by_id(obj[prop], all_objects)
                if value_obj is None or (
                    not self.check_object_type(value_obj, props[prop])
                    and not props[prop] in value_obj["simbotObjectClass"]
                ):
                    sats[prop] = False
                    continue
            elif prop == "parentReceptacles":  # list of objectIds, which don't directly reveal objectType.
                parent_receptacles = self.get_parent_receptacles(obj, all_objects)
                parent_match = False
                if parent_receptacles is not None:
                    for oid in parent_receptacles:
                        _obj = self.get_obj_by_id(oid, all_objects)
                        # value of parentReceptacle in JSON is objectType to contain.
                        if self.check_object_type(_obj, props[prop]) or props[prop] in _obj["simbotObjectClass"]:
                            parent_match = True
                if not parent_match:
                    sats[prop] = False
                    continue
            # Binary properties encoded as 1/0 truths in JSON.
            elif type(props[prop]) is int and (props[prop] == 1 or props[prop] == 0):
                if (obj[prop] and props[prop] == 0) or (not obj[prop] and props[prop] == 1):
                    sats[prop] = False
                    continue
            # Properties that return lists.
            elif type(obj[prop]) is list:
                if props[prop] not in obj[prop]:
                    sats[prop] = False
                    continue
            # Direct value comparisons.
            elif props[prop] != obj[prop]:
                sats[prop] = False
                continue
            # If we get through all these condition checks without failing, prop is satisfied.
            sats[prop] = True
        assert len(props) == len(sats)
        return sats

    @staticmethod
    def check_object_type(obj, desired_value):
        if obj["objectType"] == desired_value:
            return True
        elif (
            (obj["objectType"] == "SinkBasin" and desired_value == "Sink")
            or (obj["objectType"] == "Sink" and desired_value == "SinkBasin")
            or (obj["objectType"] == "BathtubBasin" and desired_value == "Bathtub")
            or (obj["objectType"] == "Bathtub" and desired_value == "BathtubBasin")
        ):
            return True
        else:
            return False

    def check_component_all_instances(self, all_objects_cur_state, component):
        if "task_name" in component:
            raise NotImplementedError('Determiner "all" is not supported with components that are Tasks')

        success = True
        satisfied_objects = list()
        problem_objects = list()
        all_objects = all_objects_cur_state
        for obj in all_objects:
            props_sat = self.obj_satisfies_props(obj, component["conditions"], all_objects)
            # If the object matches the primary condition, then it must satisfy properties.
            if props_sat[component["primary_condition"]]:
                props_sat_v = list(props_sat.values())
                if not np.all(props_sat_v):
                    success = False  # if any one object doesn't satisfy, the subgoal isn't satisfied
                    problem_objects.append({"object": obj, "props_sat": props_sat})
                else:
                    satisfied_objects.append(obj)

        output = dict()
        output["success"] = success
        # Total number of conditions needed in this component = Number of conditions per object * Number of object of
        # this type
        num_conditions_per_obj = len(component["conditions"].keys())
        output["goal_conditions_total"] = num_conditions_per_obj * (len(satisfied_objects) + len(problem_objects))
        output["satisfied_objects"] = satisfied_objects
        output["candidate_objects"] = [candidate["object"] for candidate in problem_objects]
        # satisfied_objects have all conditions met; for the others add the number of conditions met
        output["goal_conditions_satisfied"] = (num_conditions_per_obj * len(satisfied_objects)) + sum(
            [sum(candidate["props_sat"].values()) for candidate in problem_objects]
        )
        output["steps"] = list()
        keys_to_problem_objects = dict()
        for candidate in problem_objects:
            for key in candidate["props_sat"]:
                if not candidate["props_sat"][key] and key in component["condition_failure_descs"]:
                    if key not in keys_to_problem_objects:
                        keys_to_problem_objects[key] = list()
                    keys_to_problem_objects[key].append(candidate)
        for key, desc in component["condition_failure_descs"].items():
            if key in keys_to_problem_objects:
                for candidate in keys_to_problem_objects[key]:
                    output["steps"].append(
                        {
                            "success": False,
                            "objectId": candidate["object"]["objectId"],
                            "objectType": candidate["object"]["objectType"],
                            "desc": desc,
                        }
                    )
            else:
                representative_obj = None
                if len(satisfied_objects) > 0:
                    representative_obj = satisfied_objects[0]
                if representative_obj is not None:
                    output["steps"].append(
                        {
                            "success": True,
                            "objectId": representative_obj["objectId"],
                            "objectType": representative_obj["objectType"],
                            "desc": desc,
                        }
                    )
        output["problem_keys"] = dict()
        for candidate in problem_objects:
            output["problem_keys"][candidate["object"]["objectId"]] = list()
            for key in candidate["props_sat"]:
                if not candidate["props_sat"][key]:
                    output["problem_keys"][candidate["object"]["objectId"]].append(
                        {
                            "objectType": candidate["object"]["objectType"],
                            "determiner": component["determiner"],
                            "property_name": key,
                            "desired_property_value": component["conditions"][key],
                        }
                    )

        return output

    def check_relation(
        self,
        relation,
        per_component_satisfied_objects,
        per_component_candidate_objects,
        all_objects_cur_state,
        num_task_instances=1,
    ):
        if len(relation["tail_entity_list"]) > 1:
            raise NotImplementedError(
                "Relation checking not implemented for relations with more than one ail entity. Check definition of task"
                + str(self.task_name)
            )

        output = dict()
        # Assume one goal condition per object for which the relation is to be satisfied. Then count the number of head
        # objects to be adjusted
        output["goal_conditions_total"] = 0
        for idx, head_determiner in enumerate(relation["head_determiner_list"]):
            if head_determiner not in ["a", "all"]:
                output["goal_conditions_total"] += num_task_instances * int(head_determiner)
            elif head_determiner == "a":
                output["goal_conditions_total"] += num_task_instances
            else:
                head_entity = relation["head_entity_list"][idx]
                head_candidate_objects = list()
                if head_entity in per_component_satisfied_objects:
                    head_candidate_objects += per_component_satisfied_objects[head_entity]
                if head_entity in per_component_candidate_objects:
                    head_candidate_objects += per_component_candidate_objects[head_entity]
                output["goal_conditions_total"] += num_task_instances * len(head_candidate_objects)

        tail_determiner = relation["tail_determiner_list"][0]
        tail_candidate_objects = list()
        if relation["tail_entity_list"][0] in per_component_satisfied_objects:
            tail_candidate_objects += per_component_satisfied_objects[relation["tail_entity_list"][0]]
        if relation["tail_entity_list"][0] in per_component_candidate_objects:
            tail_candidate_objects += per_component_candidate_objects[relation["tail_entity_list"][0]]
        tail_candidate_obj_ids = set([obj["objectId"] for obj in tail_candidate_objects])
        if len(tail_candidate_obj_ids) < 1:
            output["success"] = False
            output["satisfied_objects"] = []
            output["steps"] = []
            output["problem_keys"] = []
            output["goal_conditions_satisfied"] = 0
            return output
        tail_obj_type = self.get_obj_by_id(list(tail_candidate_obj_ids)[0], all_objects_cur_state)["objectType"]
        num_head_entities = len(relation["head_entity_list"])

        steps = list()
        problem_keys = dict()
        satisfied_objects = list()

        if tail_determiner == "a":
            success = True
            goal_conditions_satisfied = 0

            for idx in range(num_head_entities):
                cur_satisfied_objects = list()
                cur_unsatisfied_objects = list()
                head_determiner = relation["head_determiner_list"][idx]
                if head_determiner == "0":
                    continue

                head_entity = relation["head_entity_list"][idx]
                head_candidate_objects = list()
                if head_entity in per_component_satisfied_objects:
                    head_candidate_objects += per_component_satisfied_objects[head_entity]
                if head_entity in per_component_candidate_objects:
                    head_candidate_objects += per_component_candidate_objects[head_entity]

                for head_obj in head_candidate_objects:
                    if relation["property"] == "parentReceptacles":
                        head_property_vals = self.get_parent_receptacles(head_obj, all_objects_cur_state)
                    else:
                        head_property_vals = head_obj[relation["property"]]
                    cur_head_satisfied = False
                    if head_property_vals is not None:
                        for property_value_obj_id in head_property_vals:
                            if property_value_obj_id in tail_candidate_obj_ids:
                                cur_head_satisfied = True
                                cur_satisfied_objects.append(head_obj)
                                break
                    if not cur_head_satisfied:
                        cur_unsatisfied_objects.append(head_obj)
                goal_conditions_satisfied += len(cur_satisfied_objects)

                if head_determiner == "all":
                    if len(cur_unsatisfied_objects) > 0:
                        for obj in cur_unsatisfied_objects:
                            steps.append(
                                {
                                    "success": False,
                                    "objectId": obj["objectId"],
                                    "objectType": obj["objectType"],
                                    "desc": relation["failure_desc"],
                                }
                            )
                            if obj["objectId"] not in problem_keys:
                                problem_keys[obj["objectId"]] = list()
                            problem_keys[obj["objectId"]].append(
                                {
                                    "objectType": obj["objectType"],
                                    "determiner": head_determiner,
                                    "property_name": relation["property"],
                                    "desired_property_value": tail_obj_type,
                                }
                            )
                        success = False
                    elif len(cur_satisfied_objects) > 0:
                        representative_obj = cur_satisfied_objects[0]
                        steps.append(
                            {
                                "success": True,
                                "objectId": representative_obj["objectId"],
                                "objectType": representative_obj["objectType"],
                                "desc": relation["failure_desc"],
                            }
                        )
                else:
                    num_instances_needed = num_task_instances
                    if head_determiner != "a":
                        num_instances_needed = num_task_instances * int(head_determiner)
                    if len(cur_satisfied_objects) < num_instances_needed:
                        success = False
                        num_unsatisfied_objects_needed = num_instances_needed - len(cur_satisfied_objects)
                        num_unsatisfied_objects_available = min(
                            num_unsatisfied_objects_needed, len(cur_unsatisfied_objects)
                        )
                        for obj in cur_unsatisfied_objects[:num_unsatisfied_objects_available]:
                            steps.append(
                                {
                                    "success": False,
                                    "objectId": obj["objectId"],
                                    "objectType": obj["objectType"],
                                    "desc": relation["failure_desc"],
                                }
                            )
                            if obj["objectId"] not in problem_keys:
                                problem_keys[obj["objectId"]] = list()
                            problem_keys[obj["objectId"]].append(
                                {
                                    "objectType": obj["objectType"],
                                    "determiner": head_determiner,
                                    "property_name": relation["property"],
                                    "desired_property_value": tail_obj_type,
                                }
                            )
                satisfied_objects += cur_satisfied_objects

            output["success"] = success
            output["satisfied_objects"] = satisfied_objects
            output["steps"] = steps
            output["problem_keys"] = problem_keys
            output["goal_conditions_satisfied"] = goal_conditions_satisfied
            if output["success"] and len(output["satisfied_objects"]) > 0 and len(output["steps"]) == 0:
                representative_obj = satisfied_objects[0]
                steps.append(
                    {
                        "success": True,
                        "objectId": representative_obj["objectId"],
                        "objectType": representative_obj["objectType"],
                        "desc": relation["failure_desc"],
                    }
                )
            return output

        elif tail_determiner == "the":
            satisfied_tail_objs = list()
            sorted_candidates_tail_objs = list()

            for tail_obj in tail_candidate_objects:
                tail_obj_id = tail_obj["objectId"]
                cur_tail_obj_status = dict()
                cur_tail_obj_status["tail_obj"] = tail_obj
                cur_tail_obj_status["per_head_status"] = list()

                for idx in range(num_head_entities):
                    cur_satisfied_objects = list()
                    cur_unsatisfied_objects = list()
                    cur_unsatisfied_descs = list()
                    cur_unsatisfied_keys = list()

                    head_determiner = relation["head_determiner_list"][idx]
                    if head_determiner == "0":
                        continue
                    head_entity = relation["head_entity_list"][idx]
                    head_candidate_objects = list()
                    if head_entity in per_component_satisfied_objects:
                        head_candidate_objects += per_component_satisfied_objects[head_entity]
                    if head_entity in per_component_candidate_objects:
                        head_candidate_objects += per_component_candidate_objects[head_entity]

                    for head_obj in head_candidate_objects:
                        if relation["property"] == "parentReceptacles":
                            head_property_vals = self.get_parent_receptacles(head_obj, all_objects_cur_state)
                        else:
                            head_property_vals = head_obj[relation["property"]]
                        if head_property_vals is not None and tail_obj_id in head_property_vals:
                            cur_satisfied_objects.append(head_obj)
                        else:
                            cur_unsatisfied_objects.append(head_obj)
                            cur_unsatisfied_descs.append([relation["failure_desc"]])
                            cur_unsatisfied_keys.append(
                                [[head_determiner, relation["property"], tail_obj["objectType"]]]
                            )

                    if head_determiner == "all":
                        instances_needed = len(head_candidate_objects)
                    elif head_determiner == "a":
                        instances_needed = 1
                    else:
                        instances_needed = int(head_determiner)

                    cur_tail_obj_status["per_head_status"].append(
                        {
                            "head_determiner": head_determiner,
                            "head_entity": head_entity,
                            "satisfied_objects": cur_satisfied_objects,
                            "unsatisfied_objects": cur_unsatisfied_objects,
                            "unsatisfied_descs": cur_unsatisfied_descs,
                            "unsatisfied_keys": cur_unsatisfied_keys,
                            "instances_needed": instances_needed,
                            "goal_conditions_satisfied": min(instances_needed, len(cur_satisfied_objects)),
                            "success": len(cur_satisfied_objects) >= instances_needed,
                        }
                    )

                success = np.all([e["success"] for e in cur_tail_obj_status["per_head_status"]])
                if success:
                    satisfied_tail_objs.append(cur_tail_obj_status)
                    if len(satisfied_tail_objs) >= num_task_instances:
                        break
                    continue

                num_heads_satisfied = sum([e["success"] for e in cur_tail_obj_status["per_head_status"]])
                instances_per_head_satisfied = [
                    min(e["instances_needed"], len(e["satisfied_objects"]))
                    for e in cur_tail_obj_status["per_head_status"]
                ]
                cur_tail_obj_status["num_heads_satisfied"] = num_heads_satisfied
                cur_tail_obj_status["num_instances_satisfied"] = sum(instances_per_head_satisfied)
                inserted = False
                for idx in range(len(sorted_candidates_tail_objs)):
                    if (
                        cur_tail_obj_status["num_heads_satisfied"]
                        > sorted_candidates_tail_objs[idx]["num_heads_satisfied"]
                        or cur_tail_obj_status["num_instances_satisfied"]
                        > sorted_candidates_tail_objs[idx]["num_instances_satisfied"]
                    ):
                        sorted_candidates_tail_objs.insert(idx, cur_tail_obj_status)
                        inserted = True
                        break
                if not inserted:
                    sorted_candidates_tail_objs.append(cur_tail_obj_status)
                num_tail_candidates_needed = max(0, num_task_instances - len(satisfied_tail_objs))
                if len(sorted_candidates_tail_objs) > num_tail_candidates_needed:
                    sorted_candidates_tail_objs = sorted_candidates_tail_objs[:num_tail_candidates_needed]

            if len(satisfied_tail_objs) == 0 and len(sorted_candidates_tail_objs) == 0:
                # This should ideally never happen - it means there are no tail candidates
                raise NotImplementedError(
                    'Not implemented - handling tail determiner "the" with no tail entity candidates'
                )
            else:
                output["success"] = len(satisfied_tail_objs) >= num_task_instances
                output["satisfied_objects"] = list()
                output["goal_conditions_satisfied"] = 0
                for idx, tail_obj in enumerate(satisfied_tail_objs):
                    for head_status in tail_obj["per_head_status"]:
                        if idx < num_task_instances:
                            output["goal_conditions_satisfied"] += head_status["goal_conditions_satisfied"]
                        output["satisfied_objects"] += head_status["satisfied_objects"]

                output["steps"] = list()
                output["problem_keys"] = dict()
                num_tail_candidates_needed = max(0, num_task_instances - len(satisfied_tail_objs))
                if num_tail_candidates_needed > 0:
                    tail_candidates = sorted_candidates_tail_objs[:num_tail_candidates_needed]
                    for tail_obj in tail_candidates:
                        for head_details in tail_obj["per_head_status"]:
                            output["goal_conditions_satisfied"] += head_details["goal_conditions_satisfied"]
                            num_problem_instances = head_details["instances_needed"] - len(
                                head_details["satisfied_objects"]
                            )
                            if num_problem_instances > 0:
                                num_problem_instances_available = min(
                                    num_problem_instances, len(head_details["unsatisfied_objects"])
                                )
                                for obj in head_details["unsatisfied_objects"][:num_problem_instances_available]:
                                    output["steps"].append(
                                        {
                                            "success": False,
                                            "objectId": obj["objectId"],
                                            "objectType": obj["objectType"],
                                            "desc": relation["failure_desc"],
                                        }
                                    )
                                    if obj["objectId"] not in output["problem_keys"]:
                                        output["problem_keys"][obj["objectId"]] = list()
                                    output["problem_keys"][obj["objectId"]].append(
                                        {
                                            "determiner": head_details["head_determiner"],
                                            "property_name": relation["property"],
                                            "desired_property_value": tail_obj["tail_obj"]["objectType"],
                                            "objectType": obj["objectType"],
                                        }
                                    )
                return output
        else:
            raise NotImplementedError(
                "No support for tail determiner: " + str(tail_determiner) + ". Supported values: a. the"
            )

    @staticmethod
    def flatten_list(input_list):
        return [item for sublist in input_list for item in sublist]

    @staticmethod
    def get_combined_problem_key_dict(list_of_dicts):
        output_dict = dict()
        for input_dict in list_of_dicts:
            for key in input_dict:
                if key in output_dict:
                    output_dict[key] += input_dict[key]
                else:
                    output_dict[key] = input_dict[key]
        return output_dict

    # Takes in output of check_episode progress and picks a reasonable object to show with this task's subgoal
    @staticmethod
    def __get_representative_obj_id(task_output):
        if (
            not task_output["success"]
            and "candidate_objects" in task_output
            and task_output["candidate_objects"] is not None
            and len(task_output["candidate_objects"]) > 0
        ):
            return task_output["candidate_objects"][0]["objectId"]
        elif (
            "satisfied_objects" in task_output
            and task_output["satisfied_objects"] is not None
            and len(task_output["satisfied_objects"]) > 0
        ):
            return task_output["satisfied_objects"][0]["objectId"]
        elif (
            "candidate_objects" in task_output
            and task_output["candidate_objects"] is not None
            and len(task_output["candidate_objects"]) > 0
        ):
            return task_output["candidate_objects"][0]["objectId"]
        else:
            last_obj_id = ""
            for subgoal in task_output["subgoals"]:
                for step in subgoal["steps"]:
                    if step["success"]:
                        return step["objectId"]
                    last_obj_id = step["objectId"]
            return last_obj_id

    def check_episode_progress(
        self, all_objects_cur_state, simulator=None, num_instances_needed=1, use_task_candidates_in_relations=False
    ):
        """
        :param all_objects_cur_state: List of dictionaries, each of which has key, value pairs corresponding to
        current properties of an object in the environment
        :param simulator: instance of Simulator_THOR or None. If set to None progress check output will not sort
        candidates by distance to agent
        :param num_instances_needed: Only relevant for tasks with task_anchor_object != None - Sets the number of anchor
        objects to be created
        :param use_task_candidates_in_relations: Set to True if relations should be checked using incomplete subtasks
        """
        self.write_task_params()

        all_subgoals = list()
        task_satisfied_objects = None
        task_candidate_objects = None
        per_component_satisfied_objects = dict()
        per_component_candidate_objects = dict()

        for component_key, component in self.components.items():
            component_instances_needed = num_instances_needed

            if component["determiner"] == "0":
                continue

            if component["determiner"] == "all":
                component_output = self.check_component_all_instances(all_objects_cur_state, component)
                component_subgoal = dict()
                component_subgoal["success"] = component_output["success"]
                component_subgoal["description"] = ""
                component_subgoal["steps"] = component_output["steps"]
                component_subgoal["problem_keys"] = component_output["problem_keys"]
                component_subgoal["goal_conditions_total"] = component_output["goal_conditions_total"]
                component_subgoal["goal_conditions_satisfied"] = component_output["goal_conditions_satisfied"]
                if not component_output["success"] or len(component_output["steps"]) > 0:
                    all_subgoals.append(component_subgoal)

                per_component_satisfied_objects[component_key] = component_output["satisfied_objects"]
                per_component_candidate_objects[component_key] = component_output["candidate_objects"]
                if self.task_anchor_object == component_key:
                    task_satisfied_objects = component_output["satisfied_objects"]
                    task_candidate_objects = component_output["candidate_objects"]

            else:
                if component["determiner"] != "a":
                    number_determiner = int(component["determiner"])
                    component_instances_needed *= number_determiner

                if "task_name" in component:
                    component_output = component["task"].check_episode_progress(
                        all_objects_cur_state,
                        simulator,
                        num_instances_needed=component_instances_needed,
                        use_task_candidates_in_relations=use_task_candidates_in_relations,
                    )

                    component_subgoal = dict()
                    component_subgoal["representative_obj_id"] = self.__get_representative_obj_id(component_output)
                    component_subgoal["step_successes"] = [
                        subgoal["success"] for subgoal in component_output["subgoals"]
                    ]
                    component_subgoal["success"] = np.all(component_subgoal["step_successes"])
                    component_subgoal["description"] = component["task"].desc
                    if component["determiner"] not in ["a", "the", "all"]:
                        component_subgoal["description"] = (
                            component["determiner"] + " x " + component_subgoal["description"]
                        )
                    component_subgoal["steps"] = self.flatten_list(
                        [
                            [step for step in subgoal["steps"] if not step["success"]]
                            for subgoal in component_output["subgoals"]
                        ]
                    )
                    component_subgoal["problem_keys"] = self.get_combined_problem_key_dict(
                        [subgoal["problem_keys"] for subgoal in component_output["subgoals"]]
                    )
                    component_subgoal["goal_conditions_total"] = component_output["goal_conditions_total"]
                    component_subgoal["goal_conditions_satisfied"] = component_output["goal_conditions_satisfied"]
                    all_subgoals.append(component_subgoal)

                    per_component_satisfied_objects[component_key] = component_output["satisfied_objects"]
                    if use_task_candidates_in_relations:
                        per_component_candidate_objects[component_key] = component_output["candidate_objects"]
                    if self.task_anchor_object == component_key:
                        task_satisfied_objects = component_output["satisfied_objects"]
                        task_candidate_objects = component_output["candidate_objects"]

                else:
                    component_output = self.check_component_n_instances(
                        all_objects_cur_state,
                        component,
                        num_instances=component_instances_needed,
                        simulator=simulator,
                        allow_state_change=False,
                    )

                    component_subgoal = dict()
                    component_subgoal["success"] = component_output["success"]
                    component_subgoal["description"] = ""
                    component_subgoal["steps"] = component_output["steps"]
                    component_subgoal["problem_keys"] = component_output["problem_keys"]
                    component_subgoal["goal_conditions_total"] = component_output["goal_conditions_total"]
                    component_subgoal["goal_conditions_satisfied"] = component_output["goal_conditions_satisfied"]
                    all_subgoals.append(component_subgoal)

                    per_component_satisfied_objects[component_key] = component_output["satisfied_objects"]
                    per_component_candidate_objects[component_key] = component_output["candidate_objects"]
                    if self.task_anchor_object == component_key:
                        task_satisfied_objects = component_output["satisfied_objects"]
                        task_candidate_objects = component_output["candidate_objects"]

        for relation in self.relations:
            relation_output = self.check_relation(
                relation,
                per_component_satisfied_objects,
                per_component_candidate_objects,
                all_objects_cur_state,
                num_instances_needed,
            )
            relation_subgoal = dict()
            relation_subgoal["success"] = relation_output["success"]
            relation_subgoal["description"] = ""
            relation_subgoal["steps"] = relation_output["steps"]
            relation_subgoal["problem_keys"] = relation_output["problem_keys"]
            relation_subgoal["goal_conditions_total"] = relation_output["goal_conditions_total"]
            relation_subgoal["goal_conditions_satisfied"] = relation_output["goal_conditions_satisfied"]
            all_subgoals.append(relation_subgoal)

        task_output = dict()
        task_output["description"] = self.desc
        task_output["success"] = np.all([subgoal["success"] for subgoal in all_subgoals])
        task_output["satisfied_objects"] = task_satisfied_objects
        task_output["candidate_objects"] = task_candidate_objects
        task_output["subgoals"] = all_subgoals
        task_output["goal_conditions_total"] = sum([subgoal["goal_conditions_total"] for subgoal in all_subgoals])
        task_output["goal_conditions_satisfied"] = sum(
            [subgoal["goal_conditions_satisfied"] for subgoal in all_subgoals]
        )

        return task_output
