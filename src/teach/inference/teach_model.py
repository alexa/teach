# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

from abc import abstractmethod
from typing import List


class TeachModel:
    @abstractmethod
    def __init__(self, process_index: int, num_processes: int, model_args: List[str]):
        """
        A model will be initialized for each evaluation process.

        See sample_model.py for a sample implementation.

        :param process_index: index of the eval process that launched the model
        :param num_processes: total number of processes launched
        :param model_args: extra CLI arguments to teach_eval will be passed along to the model
        """

    @abstractmethod
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

    @abstractmethod
    def start_new_edh_instance(self, edh_instance, edh_history_images, edh_name=None):
        """
        This method will be called at the start of each EDH instance after the environment has been set to the
        initial state by replaying history actions but before any actions are requested from the model by calling
        get_next_action
        :param edh_instance: EDH instance
        :param edh_history_images: List of images as PIL Image objects (loaded from files in
                                   edh_instance['driver_image_history'])
        :param edh_name: EDH instance file name
        """
