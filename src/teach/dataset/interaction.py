# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from collections import OrderedDict

from teach.dataset.actions import (
    Action_Audio,
    Action_Basic,
    Action_Keyboard,
    Action_MapGoal,
    Action_Motion,
    Action_ObjectInteraction,
    Action_ProgressCheck,
)


class Interaction:
    def __init__(self, agent_id, action, is_object=False, status=None, time_start=None):
        self.agent_id = agent_id
        self.action = action
        self.is_object = is_object
        self.status = status
        self.time_start = time_start

    def to_dict(self):
        _dict = OrderedDict()

        if self.is_object:
            _dict["object_id"] = self.agent_id
        else:
            _dict["agent_id"] = self.agent_id

        _dict.update(self.action.to_dict())
        if self.status is not None:
            _dict["status"] = self.status

        return _dict

    @classmethod
    def from_dict(cls, interaction_dict, action_type) -> "Interaction":

        if "object_id" in interaction_dict:
            is_object = True
            agent_id = interaction_dict["object_id"]
        else:
            is_object = False
            agent_id = interaction_dict["agent_id"]

        if action_type == "Motion":
            action = Action_Motion.from_dict(interaction_dict)
        elif action_type == "MapGoal":
            action = Action_MapGoal.from_dict(interaction_dict)
        elif action_type == "ObjectInteraction":
            action = Action_ObjectInteraction.from_dict(interaction_dict)
        elif action_type == "ProgressCheck":
            action = Action_ProgressCheck.from_dict(interaction_dict)
        elif action_type == "Keyboard":
            action = Action_Keyboard.from_dict(interaction_dict)
        elif action_type == "Audio":
            action = Action_Audio.from_dict(interaction_dict)
        else:
            action = Action_Basic.from_dict(interaction_dict)

        status = interaction_dict.get("status")
        time_start = interaction_dict.get("time_start")
        return cls(agent_id=agent_id, action=action, is_object=is_object, status=status, time_start=time_start)
