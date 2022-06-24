# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import argparse
from typing import List

import numpy as np

from teach.inference.actions import all_agent_actions, obj_interaction_actions
from teach.inference.teach_model import TeachModel
from teach.logger import create_logger

logger = create_logger(__name__)


class SampleModel(TeachModel):
    """
    Sample implementation of TeachModel.
    Demonstrates usage of custom arguments as well as sample implementation of get_next_actions method
    """

    def __init__(self, process_index: int, num_processes: int, model_args: List[str]):
        """Constructor

        :param process_index: index of the eval process that launched the model
        :param num_processes: total number of processes launched
        :param model_args: extra CLI arguments to teach_eval will be passed along to the model
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--seed", type=int, default=1, help="Random seed")
        args = parser.parse_args(model_args)

        logger.info(f"SampleModel using seed {args.seed}")
        np.random.seed(args.seed)

    def get_next_action(self, img, edh_instance, prev_action, img_name=None, edh_name=None):
        """
        This method will be called at each timestep during inference to get the next predicted action from the model.
        :param img: PIL Image containing agent's egocentric image
        :param edh_instance: EDH instance
        :param prev_action: One of None or a dict with keys 'action' and 'obj_relative_coord' containing returned values
        from a previous call of get_next_action
        :param img_name: image file name
        :param edh_name: EDH instance file name
        :return action: An action name from all_agent_actions
        :return obj_relative_coord: A relative (x, y) coordinate (values between 0 and 1) indicating an object in the image;
        The TEACh wrapper on AI2-THOR examines the ground truth segmentation mask of the agent's egocentric image, selects
        an object in a 10x10 pixel patch around the pixel indicated by the coordinate if the desired action can be
        performed on it, and executes the action in AI2-THOR.
        """
        action = np.random.choice(all_agent_actions)
        obj_relative_coord = None
        if action in obj_interaction_actions:
            obj_relative_coord = [
                np.random.uniform(high=0.99),
                np.random.uniform(high=0.99),
            ]
        return action, obj_relative_coord

    def start_new_edh_instance(self, edh_instance, edh_history_images, edh_name=None):
        """
        Since this class produces random actions at every time step, no particular setup is needed. When running model
        inference, this would be a suitable place to preprocess the dialog, action and image history
        :param edh_instance: EDH instance
        :param edh_history_images: List of images as PIL Image objects (loaded from files in
                                   edh_instance['driver_image_history'])
        :param edh_name: EDH instance file name
        """
        return True
