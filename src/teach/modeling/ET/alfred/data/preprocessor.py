import copy

import revtok
from alfred.utils import data_util
from vocab import Vocab


class Preprocessor(object):
    def __init__(self, vocab, subgoal_ann=False, is_test_split=False, frame_size=300):
        self.subgoal_ann = subgoal_ann
        self.is_test_split = is_test_split
        self.frame_size = frame_size

        if vocab is None:
            self.vocab = {
                "word": Vocab(["<<pad>>", "<<seg>>", "<<goal>>", "<<mask>>"]),
                "action_low": Vocab(["<<pad>>", "<<seg>>", "<<stop>>", "<<mask>>"]),
                "action_high": Vocab(["<<pad>>", "<<seg>>", "<<stop>>", "<<mask>>"]),
            }
        else:
            self.vocab = vocab

        self.word_seg = self.vocab["word"].word2index("<<seg>>", train=False)

    @staticmethod
    def numericalize(vocab, words, train=True):
        """
        converts words to unique integers
        """
        if not train:
            new_words = set(words) - set(vocab.counts.keys())
            if new_words:
                # replace unknown words with <<pad>>
                words = [w if w not in new_words else "<<pad>>" for w in words]
        return vocab.word2index(words, train=train)

    def process_language(self, ex, traj, r_idx, is_test_split=False):
        if self.is_test_split:
            is_test_split = True

        instr_anns = [utterance for (speaker, utterance) in ex["dialog_history"]]
        instr_anns = [revtok.tokenize(data_util.remove_spaces_and_lower(instr_ann)) for instr_ann in instr_anns]
        instr_anns = [[w.strip().lower() for w in instr_ann] for instr_ann in instr_anns]
        traj["ann"] = {
            "instr": [instr_ann + ["<<instr>>"] for instr_ann in instr_anns],
        }
        traj["ann"]["instr"] += [["<<stop>>"]]
        if "num" not in traj:
            traj["num"] = {}
        traj["num"]["lang_instr"] = [
            self.numericalize(self.vocab["word"], x, train=not is_test_split) for x in traj["ann"]["instr"]
        ]

    def tokenize_and_numericalize(self, dialog_history, numericalize=True, train=False):
        instr_anns = [utterance for (speaker, utterance) in dialog_history]

        # tokenize annotations
        instr_anns = [revtok.tokenize(data_util.remove_spaces_and_lower(instr_ann)) for instr_ann in instr_anns]

        instr_anns = [[w.strip().lower() for w in instr_ann] for instr_ann in instr_anns]
        instr = [instr_ann + ["<<instr>>"] for instr_ann in instr_anns]

        instr += [["<<stop>>"]]

        if numericalize:
            instr = [self.numericalize(self.vocab["word"], word, train=train) for word in instr]
        instr = sum(instr, [])  # flatten
        return instr

    def process_actions(self, ex, traj):
        if "num" not in traj:
            traj["num"] = {}
        traj["num"]["driver_actions_low"] = list()
        traj["num"]["driver_actions_pred_mask"] = list()
        for action in ex["driver_action_history"]:
            action_dict_with_idx = copy.deepcopy(action)
            action_dict_with_idx["action"] = (self.vocab["action_low"].word2index(action["action_name"], train=True),)
            traj["num"]["driver_actions_low"].append(action_dict_with_idx)
            traj["num"]["driver_actions_pred_mask"].append(0)
        for action in ex["driver_actions_future"]:
            action_dict_with_idx = copy.deepcopy(action)
            action_dict_with_idx["action"] = (self.vocab["action_low"].word2index(action["action_name"], train=True),)
            traj["num"]["driver_actions_low"].append(action_dict_with_idx)
            traj["num"]["driver_actions_pred_mask"].append(1)
