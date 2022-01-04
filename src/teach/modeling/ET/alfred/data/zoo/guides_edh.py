import logging
import os

import torch
from alfred import constants
from alfred.data.zoo.base import BaseDataset

from teach.logger import create_logger

logger = create_logger(__name__, level=logging.INFO)


class GuidesEdhDataset(BaseDataset):
    def __init__(self, name, partition, args, ann_type):
        super().__init__(name, partition, args, ann_type)
        # preset values
        self._load_features = True
        self._load_frames = True
        # load the vocabulary for object classes
        vocab_obj_file = os.path.join(constants.ET_ROOT, constants.OBJ_CLS_VOCAB)
        logger.info("Loading object vocab from %s" % vocab_obj_file)
        self.vocab_obj = torch.load(vocab_obj_file)

    def load_data(self, path):
        return super().load_data(path, feats=True, jsons=True)

    def __getitem__(self, idx):
        task_json, key = self.jsons_and_keys[idx]
        feat_dict = {}
        if self._load_features:
            feat_dict = self.load_features(task_json)
        if self._load_frames:
            feat_dict["frames"] = self.load_frames(key)

        # Add a stop action and duplicate the last frame
        feat_dict["action"].append(self.vocab_out.word2index("Stop"))
        feat_dict["frames"] = torch.cat((feat_dict["frames"], torch.unsqueeze(feat_dict["frames"][-1, :], 0)), 0)
        feat_dict["obj_interaction_action"].append(0)
        feat_dict["driver_actions_pred_mask"].append(0)

        if self.args.no_lang:
            feat_dict["lang"] = [self.vocab_in.word2index("<<pad>>")]
        elif self.args.no_vision:
            feat_dict["frames"] = torch.rand(feat_dict["frames"].shape)

        return task_json, feat_dict

    def load_features(self, task_json):
        """
        load features from task_json
        """
        feat = dict()
        # language inputs
        feat["lang"] = GuidesEdhDataset.load_lang(task_json)

        # action outputs
        if not self.test_mode:
            # low-level action
            feat["action"] = GuidesEdhDataset.load_action(task_json, self.vocab_out)
            feat["obj_interaction_action"] = [
                a["obj_interaction_action"] for a in task_json["num"]["driver_actions_low"]
            ]
            feat["driver_actions_pred_mask"] = task_json["num"]["driver_actions_pred_mask"]
            feat["object"] = self.load_object_classes(task_json, self.vocab_obj)

        return feat

    @staticmethod
    def load_lang(task_json):
        """
        load numericalized language from task_json
        """
        return sum(task_json["num"]["lang_instr"], [])

    @staticmethod
    def load_action(task_json, vocab_orig, action_type="action_low"):
        """
        load action as a list of tokens from task_json
        """
        if action_type == "action_low":
            # load low actions
            lang_action = [[vocab_orig.word2index(a["action_name"]) for a in task_json["num"]["driver_actions_low"]]]
            lang_action = sum(lang_action, [])
        elif action_type == "action_high_future":
            if "future_subgoals" in task_json:
                lang_action = [vocab_orig.word2index(w) for w in task_json["future_subgoals"]]
            else:
                lang_action = [0]
        elif action_type == "action_high_all":
            lang_action = [
                vocab_orig.word2index(w) for w in task_json["history_subgoals"] + task_json["future_subgoals"]
            ]
        else:
            raise NotImplementedError("Unknown action_type {}".format(action_type))
        return lang_action

    def load_object_classes(self, task_json, vocab=None):
        """
        load object classes for interactive actions
        """
        object_classes = []
        for idx, action in enumerate(task_json["num"]["driver_actions_low"]):
            if self.args.compute_train_loss_over_history or task_json["num"]["driver_actions_pred_mask"][idx] == 1:
                if action["oid"] is not None:
                    object_class = action["oid"].split("|")[0]
                    object_classes.append(object_class if vocab is None else vocab.word2index(object_class))
        return object_classes
