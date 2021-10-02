# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from collections import OrderedDict

from teach.dataset.episode import Episode


class Task:
    def __init__(self, task_id, task_name, task_nparams, task_params, subgoals, comments="", episodes=None):
        self.task_id = task_id
        self.task_name = task_name
        self.task_nparams = task_nparams
        self.task_params = task_params
        self.subgoals = subgoals
        self.comments = comments
        self.episodes = [] if episodes is None else episodes

    def add_episode(self, episode):
        self.episodes.append(episode)

    def to_dict(self):
        _dict = OrderedDict()
        _dict["task_id"] = self.task_id
        _dict["task_name"] = self.task_name
        _dict["task_params"] = self.task_params
        _dict["task_nparams"] = self.task_nparams
        _dict["subgoals"] = self.subgoals
        _dict["comments"] = self.comments
        _dict["episodes"] = [x.to_dict() for x in self.episodes]
        return _dict

    @classmethod
    def from_dict(cls, task_dict, definitions, process_init_state=True) -> "Task":
        episodes = [
            Episode.from_dict(episode_dict, definitions, process_init_state)
            for episode_dict in task_dict.get("episodes")
        ]
        return cls(
            task_id=task_dict["task_id"],
            task_name=task_dict["task_name"],
            task_nparams=task_dict["task_nparams"],
            task_params=task_dict["task_params"],
            subgoals=task_dict["subgoals"],
            comments=task_dict.get("comments"),
            episodes=episodes,
        )
