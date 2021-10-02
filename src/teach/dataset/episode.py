# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from collections import OrderedDict

from teach.dataset.initialization import Initialization
from teach.dataset.interaction import Interaction


class Episode:
    def __init__(self, episode_id, world, world_type, commander_embodied, initial_state=None, interactions=None):
        self.episode_id = episode_id
        self.world = world
        self.world_type = world_type
        self.commander_embodied = commander_embodied
        self.initial_state = initial_state
        self.interactions = interactions if interactions is not None else []
        self.final_state = None

    def reset_initial_state(self, initialization):
        self.initialization = initialization

    def add_interaction(self, interaction):
        self.interactions.append(interaction)

    def remove_interaction(self):
        if len(self.interactions) > 0:
            del self.interactions[-1]

    def to_dict(self):
        _dict = OrderedDict()
        _dict["episode_id"] = self.episode_id
        _dict["world"] = self.world
        _dict["world_type"] = self.world_type
        _dict["commander_embodied"] = str(self.commander_embodied)

        if self.initial_state is not None:
            _dict["initial_state"] = self.initial_state.to_dict()

        _dict["interactions"] = [x.to_dict() for x in self.interactions]

        if self.final_state is not None:
            _dict["final_state"] = self.final_state.to_dict()

        return _dict

    @classmethod
    def from_dict(cls, episode_dict, definitions, process_init_state=True) -> "Episode":
        interactions = []
        for interaction_dict in episode_dict.get("interactions"):
            action_type = definitions.map_actions_id2info[interaction_dict["action_id"]]["action_type"]
            interaction = Interaction.from_dict(interaction_dict, action_type)
            interactions.append(interaction)

        return cls(
            episode_dict["episode_id"],
            episode_dict["world"],
            episode_dict["world_type"],
            episode_dict["commander_embodied"],
            initial_state=Initialization.from_dict(episode_dict["initial_state"])
            if process_init_state
            else episode_dict["initial_state"],
            interactions=interactions,
        )
