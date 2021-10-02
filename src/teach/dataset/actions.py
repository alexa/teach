# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from collections import OrderedDict

from teach.dataset.pose import Pose


class Action_Basic:
    def __init__(self, action_id, time_start, duration, success=None):
        self.action_id = action_id
        self.time_start = time_start
        self.duration = duration
        self.action_type = "Motion"
        self.success = success  # whether the action was successfully carried out in the simulator [0, 1]

    def to_dict(self):
        _dict = OrderedDict()
        _dict["action_id"] = self.action_id
        _dict["time_start"] = self.time_start
        _dict["duration"] = self.duration
        _dict["success"] = self.success

        return _dict

    @classmethod
    def from_dict(cls, action_dict) -> "Action_Basic":
        return cls(
            action_id=action_dict["action_id"],
            time_start=action_dict["time_start"],
            duration=action_dict["duration"],
            success=action_dict["success"],
        )


class Action_Motion(Action_Basic):
    def __init__(self, action_id, time_start, duration, pose, pose_delta=None, success=None):
        super().__init__(action_id=action_id, time_start=time_start, duration=duration, success=success)
        self.action_type = "Motion"
        self.pose = pose
        self.pose_delta = pose_delta

    def to_dict(self):
        _dict = OrderedDict()
        _dict.update(super().to_dict())
        if self.pose_delta is not None:
            _dict["pose_delta"] = self.pose_delta.to_dict()["pose"]

        _dict.update(self.pose.to_dict())

        return _dict

    @classmethod
    def from_dict(cls, action_dict) -> "Action_Motion":
        pose_delta = action_dict.get("pose_delta")
        if pose_delta is not None:
            pose_delta = Pose.from_array(pose_delta)

        return cls(
            action_id=action_dict["action_id"],
            time_start=action_dict["time_start"],
            duration=action_dict["duration"],
            success=action_dict["success"],
            pose=Pose.from_array(action_dict["pose"]),
            pose_delta=pose_delta,
        )


class Action_MapGoal(Action_Basic):
    def __init__(self, action_id, time_start, duration, start_x, start_y, end_x, end_y, success=None):
        super().__init__(action_id=action_id, time_start=time_start, duration=duration, success=success)
        self.action_type = "MapGoal"
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

    def to_dict(self):
        _dict = OrderedDict()
        _dict.update(super().to_dict())
        _dict["start_x"] = self.start_x
        _dict["start_y"] = self.start_y
        _dict["end_x"] = self.end_x
        _dict["end_y"] = self.end_y

        return _dict

    @classmethod
    def from_dict(cls, action_dict) -> "Action_MapGoal":
        return cls(
            action_id=action_dict["action_id"],
            time_start=action_dict["time_start"],
            duration=action_dict["duration"],
            success=action_dict["success"],
            start_x=action_dict["start_x"],
            start_y=action_dict["start_y"],
            end_x=action_dict["end_x"],
            end_y=action_dict["end_y"],
        )


class Action_ObjectInteraction(Action_Basic):
    def __init__(self, action_id, time_start, duration, x, y, success=None, oid=None):
        super().__init__(action_id=action_id, time_start=time_start, duration=duration, success=success)
        self.action_type = "ObjectInteraction"
        self.x = x
        self.y = y
        self.oid = oid  # The simulator id of the object interacted with at (x, y), if any

    def to_dict(self):
        _dict = OrderedDict()
        _dict.update(super().to_dict())
        _dict["x"] = self.x
        _dict["y"] = self.y
        _dict["oid"] = self.oid

        return _dict

    @classmethod
    def from_dict(cls, action_dict) -> "Action_ObjectInteraction":
        return cls(
            action_id=action_dict["action_id"],
            time_start=action_dict["time_start"],
            duration=action_dict["duration"],
            success=action_dict["success"],
            x=action_dict["x"],
            y=action_dict["y"],
            oid=action_dict["oid"],
        )


class Action_CameraChange(Action_Basic):
    def __init__(self, action_id, time_start, duration, success=None):
        super().__init__(action_id=action_id, time_start=time_start, duration=duration, success=success)
        self.action_type = "CameraChange"

    def to_dict(self):
        _dict = OrderedDict()
        _dict.update(super().to_dict())

        return _dict

    @classmethod
    def from_dict(cls, action_dict) -> "Action_CameraChange":
        return cls(
            action_id=action_dict["action_id"],
            time_start=action_dict["time_start"],
            duration=action_dict["duration"],
            success=action_dict["success"],
        )


class Action_ProgressCheck(Action_Basic):
    def __init__(self, action_id, time_start, duration, query, success=None):
        super().__init__(action_id=action_id, time_start=time_start, duration=duration, success=success)
        self.action_type = "ProgressCheck"
        self.query = query  # either an oid (for 'SelectOid') or a search query (for 'SearchObject')

    def to_dict(self):
        _dict = OrderedDict()
        _dict.update(super().to_dict())
        _dict["query"] = self.query

        return _dict

    @classmethod
    def from_dict(cls, action_dict) -> "Action_ProgressCheck":
        return cls(
            action_id=action_dict["action_id"],
            time_start=action_dict["time_start"],
            duration=action_dict["duration"],
            query=action_dict["query"],
            success=action_dict["success"],
        )


class Action_Keyboard(Action_Basic):
    def __init__(self, action_id, time_start, duration, utterance, success=None):
        super().__init__(action_id=action_id, time_start=time_start, duration=duration, success=success)
        self.action_type = "Keyboard"
        self.utterance = utterance

    def to_dict(self):
        _dict = OrderedDict()
        _dict.update(super().to_dict())
        _dict["utterance"] = self.utterance

        return _dict

    @classmethod
    def from_dict(cls, action_dict) -> "Action_Keyboard":
        return cls(
            action_id=action_dict["action_id"],
            time_start=action_dict["time_start"],
            duration=action_dict["duration"],
            success=action_dict["success"],
            utterance=action_dict["utterance"],
        )


class Action_Audio(Action_Keyboard):
    def __init__(self, action_id, time_start, duration, utterance, file_name=None, success=None):
        super().__init__(
            action_id=action_id, time_start=time_start, duration=duration, utterance=utterance, success=success
        )
        self.action_type = "Audio"
        self.file_name = file_name

    def to_dict(self):
        _dict = OrderedDict()
        _dict.update(super().to_dict())

        if self.file_name is not None:
            _dict["file_name"] = self.file_name

        return _dict

    @classmethod
    def from_dict(cls, action_dict) -> "Action_Audio":
        return cls(
            action_id=action_dict["action_id"],
            time_start=action_dict["time_start"],
            duration=action_dict["duration"],
            success=action_dict["success"],
            utterance=action_dict["utterance"],
            file_name=action_dict["file_name"],
        )
