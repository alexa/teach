# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


import copy
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from teach.dataset.task_THOR import Task_THOR
from teach.logger import create_logger

logger = create_logger(__name__)


def load_json(filename):
    with open(filename) as h:
        return json.load(h)


def save_json(obj, filename):
    with open(filename, 'w') as h:
        json.dump(obj, h)


def reduce_float_precision(input_entry, num_places_to_retain=4, keys_to_exclude=None):
    if keys_to_exclude is None:
        keys_to_exclude = ["time_start"]

    if issubclass(type(input_entry), dict):
        output_dict = copy.deepcopy(input_entry)
        for k, v in input_entry.items():
            if k in keys_to_exclude:
                output_dict[k] = v
            elif type(v) in [dict, list]:
                output_dict[k] = reduce_float_precision(
                    v, num_places_to_retain=num_places_to_retain, keys_to_exclude=keys_to_exclude
                )
            elif type(v) == float:
                output_dict[k] = round(v, num_places_to_retain)
            else:
                output_dict[k] = v
        return output_dict

    if type(input_entry) == list:
        output_list = list()
        for v in input_entry:
            if type(v) in [dict, list]:
                output_list.append(
                    reduce_float_precision(
                        v, num_places_to_retain=num_places_to_retain, keys_to_exclude=keys_to_exclude
                    )
                )
            elif type(v) == float:
                output_list.append(round(v, num_places_to_retain))
            else:
                output_list.append(v)
        return output_list

    raise NotImplementedError("Cannot handle input of type" + str(type(input_entry)))


def are_prop_values_equal(init_value, final_value):
    if type(init_value) != type(final_value):
        return False
    elif issubclass(type(init_value), list):
        if len(init_value) != len(final_value):
            return False
        for idx in range(len(init_value)):
            if not are_prop_values_equal(init_value[idx], final_value[idx]):
                return False
    elif issubclass(type(init_value), dict):
        if len(init_value) != len(final_value):
            return False
        for key in final_value:
            if key not in init_value or not are_prop_values_equal(init_value[key], final_value[key]):
                return False
    elif type(init_value) == float:
        if not np.isclose(init_value, final_value):
            return False
    elif init_value != final_value:
        return False

    return True


def get_obj_type_from_oid(oid):
    parts = oid.split("|")
    if len(parts) == 4:
        return parts[0]
    else:
        return parts[-1].split("_")[0]


def get_state_changes(init_state, final_state):
    agent_changes = dict()
    for idx in range(len(final_state["agents"])):
        agent_init = init_state["agents"][idx]
        agent_final = init_state["agents"][idx]
        if agent_init == agent_final:
            continue
        agent_changes[idx] = dict()
        for prop in agent_final.keys():
            if prop not in agent_init or not are_prop_values_equal(agent_init, agent_final):
                agent_changes[idx][prop] = agent_final[prop]
        if len(agent_changes[idx]) == 0:
            del agent_changes[idx]

    init_obj_dict = dict()
    for obj in init_state["objects"]:
        init_obj_dict[obj["objectId"]] = obj
        if obj["objectId"] in init_state["custom_object_metadata"]:
            init_obj_dict[obj["objectId"]].update(init_state["custom_object_metadata"][obj["objectId"]])
    final_obj_dict = dict()
    for obj in final_state["objects"]:
        final_obj_dict[obj["objectId"]] = obj
        if obj["objectId"] in final_state["custom_object_metadata"]:
            final_obj_dict[obj["objectId"]].update(final_state["custom_object_metadata"][obj["objectId"]])

    init_obj_id_given_final_obj_id = dict()
    for obj_id in final_obj_dict.keys():
        if obj_id in init_obj_dict.keys():
            init_obj_id_given_final_obj_id[obj_id] = obj_id
        elif len(obj_id.split("|")) > 4:
            init_obj_id_given_final_obj_id[obj_id] = "|".join(obj_id.split("|")[:4])
        else:
            init_obj_id_given_final_obj_id[obj_id] = obj_id

    obj_changes = dict()
    for object_id, obj_final in final_obj_dict.items():
        obj_init = init_obj_dict[init_obj_id_given_final_obj_id[object_id]]
        if obj_init == obj_final:
            continue
        obj_changes[object_id] = dict()
        for prop in obj_final.keys():
            if prop not in obj_init or not are_prop_values_equal(obj_init[prop], obj_final[prop]):
                obj_changes[object_id][prop] = obj_final[prop]
        if len(obj_changes[object_id]) == 0:
            del obj_changes[object_id]

    return {"agents": agent_changes, "objects": obj_changes}


def get_state_diff_changes(init_state_diff, final_state_diff):
    agent_changes = dict()
    for agent_id in final_state_diff["agents"]:
        agent_init = init_state_diff["agents"][agent_id]
        agent_final = final_state_diff["agents"][agent_id]
        if agent_init == agent_final:
            continue
        agent_changes[agent_id] = dict()
        for prop in agent_final.keys():
            if prop not in agent_init or not are_prop_values_equal(agent_init, agent_final):
                agent_changes[agent_id][prop] = agent_final[prop]
        if len(agent_changes[agent_id]) == 0:
            del agent_changes[agent_id]

    props_to_check = {
        "isToggled",
        "isBroken",
        "isFilledWithLiquid",
        "isDirty",
        "isUsedUp",
        "isCooked",
        "isOpen",
        "isPickedUp",
        "objectType",
        "simbotLastParentReceptacle",
        "simbotIsCooked",
        "simbotIsFilledWithWater",
        "simbotIsBoiled",
        "simbotIsFilledWithCoffee",
        "simbotPickedUp",
    }
    init_obj_dict = dict()
    final_obj_dict = dict()
    for obj_id, obj in init_state_diff["objects"].items():
        new_obj = dict([(k, v) for k, v in obj.items() if k in props_to_check])
        if len(new_obj.keys()) > 0:
            init_obj_dict[obj_id] = new_obj
    for obj_id, obj in final_state_diff["objects"].items():
        new_obj = dict([(k, v) for k, v in obj.items() if k in props_to_check])
        if len(new_obj.keys()) > 0:
            final_obj_dict[obj_id] = new_obj

    init_obj_id_given_final_obj_id = dict()
    obj_ids = list(set(final_obj_dict.keys()).union(init_obj_dict.keys()))
    for obj_id in obj_ids:
        if obj_id in init_obj_dict and obj_id in final_obj_dict:
            init_obj_id_given_final_obj_id[obj_id] = obj_id
        elif len(obj_id.split("|")) > 4 and "Basin" not in obj_id:
            init_obj_id_given_final_obj_id[obj_id] = "|".join(obj_id.split("|")[:4])
        else:
            init_obj_id_given_final_obj_id[obj_id] = obj_id

    obj_changes = dict()
    for object_id, obj_final in final_obj_dict.items():
        if init_obj_id_given_final_obj_id[object_id] not in init_obj_dict:
            # This object was not modified at start of EDH instance but was modified at the end => all changes are from
            # the instance
            obj_changes[object_id] = obj_final
            continue
        obj_init = init_obj_dict[init_obj_id_given_final_obj_id[object_id]]
        if obj_init == obj_final:
            continue
        obj_changes[object_id] = dict()
        for prop in obj_final.keys():
            if prop not in obj_init or not are_prop_values_equal(obj_init[prop], obj_final[prop]):
                obj_changes[object_id][prop] = obj_final[prop]
        if len(obj_changes[object_id]) == 0:
            del obj_changes[object_id]

    return {"agents": agent_changes, "objects": obj_changes}


def apply_state_diff(state, state_diff):
    for agent_id in state_diff["agents"]:
        for prop in state_diff["agents"][agent_id]:
            state["agents"][agent_id][prop] = state_diff["agents"][agent_id][prop]
    obj_changes_applied = set()
    for obj in state["objects"]:
        if obj["objectId"] in state_diff["objects"]:
            obj.update(state_diff["objects"][obj["objectId"]])
            obj_changes_applied.add(obj["objectId"])
    logger.debug("Applied changes to objects: " + str(obj_changes_applied))

    # Find objects whose changes have not been applied - these should be due to slicing or cracking
    obj_changes_remaining = set(state_diff["objects"].keys()).difference(obj_changes_applied)
    logger.debug("Objects to be changed that involved slicing / cracking :" + str(obj_changes_remaining))
    objs_to_delete = list()
    obj_idxs_to_delete = list()
    for obj_id in obj_changes_remaining:
        base_obj_id = "|".join(obj_id.split("|")[:4])
        base_obj_idx, base_obj = [
            (idx, obj) for idx, obj in enumerate(state["objects"]) if obj["objectId"] == base_obj_id
        ][0]
        new_obj = copy.deepcopy(base_obj)
        new_obj.update(state_diff["objects"][obj_id])
        logger.debug(
            "Created " + str(new_obj) + " from " + str(base_obj) + " with changes " + str(state_diff["objects"][obj_id])
        )
        state["objects"].append(new_obj)
        objs_to_delete.append(base_obj)
        obj_idxs_to_delete.append(base_obj_idx)

    obj_idxs_to_delete = set(obj_idxs_to_delete)
    logger.debug("Indices to delete:" + str(obj_idxs_to_delete))
    logger.debug("Cur objects :" + str([(idx, obj["objectId"]) for (idx, obj) in enumerate(state["objects"])]))

    # Delete unchanged versions of sliced / cracked objects
    state["objects"] = [obj for (idx, obj) in enumerate(state["objects"]) if idx not in obj_idxs_to_delete]
    logger.debug("Objects after deletion :" + str([obj["objectId"] for obj in state["objects"]]))
    return state


def create_task_thor_from_state_diff(state_diff):
    components = dict()
    props_to_check = {
        "isToggled",
        "isBroken",
        "isFilledWithLiquid",
        "isDirty",
        "isUsedUp",
        "isCooked",
        "isOpen",
        "isPickedUp",
        "objectType",
        "simbotLastParentReceptacle",
        "simbotIsCooked",
        "simbotIsFilledWithWater",
        "simbotIsBoiled",
        "simbotIsFilledWithCoffee",
        "simbotPickedUp",
    }

    # The following is to ensure that we're not checking duplicate properties which would result in biases in goal
    # condition scores
    prop_overrides = {
        "simbotPickedUp": ["isPickedUp"],
        "simbotIsCooked": ["isCooked"],
        "simbotIsFilledWithWater": ["isFilledWithLiquid"],
        "simbotIsFilledWithCoffee": ["isFilledWithLiquid"],
        "simbotIsBoiled": ["isCooked", "simbotIsCooked"],
    }

    logger.debug("Creating task from state diff ...")
    for obj_id in state_diff["objects"]:
        obj_type = get_obj_type_from_oid(obj_id)
        props_for_task = set(state_diff["objects"][obj_id].keys()).intersection(props_to_check)
        overridden_props = [
            prop_overrides[prop]
            for prop in set(state_diff["objects"][obj_id].keys()).intersection(prop_overrides.keys())
        ]
        overridden_props_flat = set([prop for prop_list in overridden_props for prop in prop_list])
        props_for_task = props_for_task.difference(overridden_props_flat)

        for prop in props_for_task:
            val = state_diff["objects"][obj_id][prop]
            # Note that creating a component for each (obj_type, prop, val) combo is fine because task checking will
            # allow objects to be shared across components
            key = str((obj_type, prop, val))
            if key in components:
                components[key]["determiner"] = int(components[key]["determiner"]) + 1
            else:
                components[key] = dict()
                components[key]["determiner"] = 1
                components[key]["primary_condition"] = "objectType"
                components[key]["instance_shareable"] = False
                components[key]["conditions"] = dict()
                components[key]["conditions"]["objectType"] = obj_type
                components[key]["condition_failure_descs"] = dict()
                components[key]["condition_failure_descs"][prop] = (
                    str(prop) + " needs to be " + str(val) + " for " + str(obj_type)
                )
                if prop == "simbotLastParentReceptacle" and val is not None:
                    components[key]["conditions"][prop] = val.split("|")[0]
                else:
                    components[key]["conditions"][prop] = val

    return Task_THOR(
        task_id=0,
        task_name="edh_custom",
        task_nparams=0,
        task_params=[],
        task_anchor_object=None,
        desc="custom EDH task",
        components=components,
        relations=[],
    )


def save_dict_as_json(data: dict, filepath: str):
    filepath = Path(filepath)

    try:
        filepath.parent.mkdir(exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create directory: {e}", exc_info=True)
        raise e

    try:
        with filepath.open(mode="w") as fp:
            json.dump(data, fp)
    except OSError as e:
        logger.error(f"Could not write file: {e}", exc_info=True)
        raise e


def with_retry(fn, retries, check_first_return_value=True):
    """
    Tries to run the given function upto retries + 1 many times in the event it raises an exception.

    :param fn: The function to run.
    :param retries: We perform this many retries in case the function fails to run successfully.
    :param check_first_return_value: We check if the first return value of the function is falsy, if it is we also retry.
    :returns: the output of fn
    :raises Exception: when all retries fail, the last caught exception is raised
    """
    max_invocations = retries + 1
    invocation_count = 0
    last_exception = None

    while invocation_count < max_invocations:
        invocation_count += 1
        try:
            output = fn()

            if check_first_return_value:
                status, *rest = output
                if not status:
                    raise Exception("the function's first return value indicated failure")

            return output
        except Exception as e:
            last_exception = e

    raise last_exception


def dynamically_load_class(package_path, class_name):
    """
    Dynamically load the specified class.
    :param package_path: Path to the package to load
    :param class_name: Name of the class within the package
    :return: the instantiated class object
    """
    module = __import__(package_path, fromlist=[class_name])
    klass = getattr(module, class_name)
    return klass


def load_images(image_dir, image_file_names):
    images = list()
    if not image_file_names:
        return images
    if not os.path.exists(image_dir):
        raise Exception(f"{image_dir} doesn't exist")
    for f in image_file_names:
        image_file = os.path.join(image_dir, f)
        if not os.path.exists(image_file):
            continue
        image_orig = Image.open(image_file)
        images.append(image_orig.copy())
        image_orig.close()
    return images


def update_objs_with_custom_metadata(objs_list, objs_metadata_dict):
    for obj in objs_list:
        if obj["objectId"] in objs_metadata_dict:
            obj.update(objs_metadata_dict[obj["objectId"]])
    return objs_list