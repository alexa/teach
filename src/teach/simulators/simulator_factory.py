# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from teach.simulators.simulator_THOR import SimulatorTHOR


class SimulatorFactory:
    def __init__(self):
        """
        simulators in the factory
        """
        self._simulators = {}

    def register(self, simulator_name, simulator):
        """
        register simulator by name
        """
        self._simulators[simulator_name] = simulator

    def create(self, simulator_name, **kwargs):
        """
        get simulator by name and initialize it with kwargs
        """
        simulator = self._simulators.get(simulator_name)
        if not simulator:
            raise ValueError(simulator_name)
        return simulator(**kwargs)


factory = SimulatorFactory()
factory.register("thor", SimulatorTHOR)
