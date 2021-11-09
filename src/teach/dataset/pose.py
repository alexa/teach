# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from collections import OrderedDict


class Pose:
    def __init__(self, x, y, z, x_rot, y_rot, z_rot):
        self.x = x
        self.y = y
        self.z = z
        self.x_rot = x_rot
        self.y_rot = y_rot
        self.z_rot = z_rot

    def to_dict(self):
        _dict = OrderedDict()
        _dict["pose"] = [self.x, self.y, self.z, self.x_rot, self.y_rot, self.z_rot]
        return _dict

    @classmethod
    def from_array(cls, pose_array) -> "Pose":
        return cls(
            x=pose_array[0],
            y=pose_array[1],
            z=pose_array[2],
            x_rot=pose_array[3],
            y_rot=pose_array[4],
            z_rot=pose_array[5],
        )


class Pose_With_ID:
    def __init__(self, identity, pose, is_object=False):
        self.identity = identity
        self.pose = pose
        self.is_object = is_object

    def to_dict(self):
        _dict = OrderedDict()

        if self.is_object:
            _dict["object_id"] = self.identity
        else:
            _dict["agent_id"] = self.identity
        _dict.update(self.pose.to_dict())
        return _dict

    @classmethod
    def from_dict(cls, pose_with_id_dict) -> "Pose_With_ID":
        is_object = False
        if "object_id" in pose_with_id_dict:
            is_object = True
            identity = pose_with_id_dict["object_id"]
        elif "thirdPartyCameraId" in pose_with_id_dict:
            identity = pose_with_id_dict["thirdPartyCameraId"]
        elif "agent_id" in pose_with_id_dict:
            identity = pose_with_id_dict["agent_id"]
        else:
            identity = 1

        if "pose" in pose_with_id_dict:
            pose = Pose.from_array(pose_with_id_dict["pose"])
        else:
            pose = Pose.from_array(
                list(pose_with_id_dict["position"].values()) + list(pose_with_id_dict["rotation"].values())
            )

        return cls(identity=identity, pose=pose, is_object=is_object)
