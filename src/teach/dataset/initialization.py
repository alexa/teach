# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from collections import OrderedDict

from teach.dataset.pose import Pose_With_ID


class Initialization:
    def __init__(self, time_start, agents=None, objects=None, custom_object_metadata=None):
        self.time_start = time_start
        self.agents = agents if agents is not None else []
        self.objects = objects if objects is not None else []
        self.custom_object_metadata = custom_object_metadata if custom_object_metadata is not None else {}

    def add_agent(self, agent):
        self.agents.append(agent)

    def add_object(self, obj):
        self.objects.append(obj)

    def reset_time(self, time_desired=0):
        # Note: We could Unix time or any desired time instead of 0
        self.time_start = time_desired

    def to_dict(self):
        _dict = OrderedDict()
        _dict["time_start"] = self.time_start

        if len(self.agents) > 0:
            _dict["agents"] = [x if type(x) is dict else x.to_dict() for x in self.agents]

        if len(self.objects) > 0:
            _dict["objects"] = [x if type(x) is dict else x.to_dict() for x in self.objects]

        if self.custom_object_metadata is not None:
            _dict["custom_object_metadata"] = self.custom_object_metadata

        return _dict

    @classmethod
    def from_dict(cls, initialization_dict) -> "Initialization":
        agents = []
        objects = []

        if "agents" in initialization_dict:
            agents = [Pose_With_ID.from_dict(x) for x in initialization_dict["agents"]]

        if "objects" in initialization_dict:
            objects = [Pose_With_ID.from_dict(x) for x in initialization_dict["objects"]]

        return cls(time_start=initialization_dict["time_start"], agents=agents, objects=objects)
