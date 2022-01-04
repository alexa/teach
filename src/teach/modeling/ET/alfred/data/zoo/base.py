import logging
import os
import pickle
import warnings

import lmdb
import numpy as np
import torch
from alfred import constants
from alfred.utils import data_util
from torch.utils.data import Dataset as TorchDataset

from teach.logger import create_logger

logger = create_logger(__name__, level=logging.INFO)


class BaseDataset(TorchDataset):
    def __init__(self, name, partition, args, ann_type):
        logger.debug("Dataset __init__ with args %s" % str(args))
        path = os.path.join(constants.ET_DATA, name)
        self.partition = partition
        self.name = name
        self.args = args
        if ann_type not in ("lang", "frames", "lang_frames"):
            raise ValueError("Unknown annotation type: {}".format(ann_type))
        self.ann_type = ann_type
        self.test_mode = False
        self.pad = constants.PAD

        # read information about the dataset
        self.dataset_info = data_util.read_dataset_info(name)
        if self.dataset_info["visual_checkpoint"]:
            logger.info("Visual checkpoint for data preprocessing: %s" % str(self.dataset_info["visual_checkpoint"]))

        # load data
        self._length = self.load_data(path)
        if self.args.fast_epoch:
            self._length = 16
        logger.info("%s dataset size = %d" % (partition, self._length))

        # load vocabularies for input language and output actions
        vocab = data_util.load_vocab(name, ann_type)
        self.vocab_in = vocab["word"]
        out_type = "action_low" if args.model == "transformer" else "action_high"
        self.vocab_out = vocab[out_type]
        logger.debug("Loaded vocab_out: %s" % str(self.vocab_out.to_dict()["index2word"]))
        # if several datasets are used, we will translate outputs to this vocab later
        self.vocab_translate = None

    def load_data(self, path, feats=True, jsons=True):
        """
        load data
        """
        # do not open the lmdb database open in the main process, do it in each thread
        if feats:
            self.feats_lmdb_path = os.path.join(path, self.partition, "feats")

        # load jsons with pickle and parse them
        if jsons:
            with open(os.path.join(path, self.partition, "jsons.pkl"), "rb") as jsons_file:
                jsons = pickle.load(jsons_file)
            self.jsons_and_keys = []
            for idx in range(len(jsons)):
                key = "{:06}".format(idx).encode("ascii")
                if key in jsons:
                    task_jsons = jsons[key]
                    for json in task_jsons:
                        # compatibility with the evaluation
                        if "task" in json and isinstance(json["task"], str):
                            pass
                        else:
                            json["task"] = "/".join(json["root"].split("/")[-3:-1])
                        # add dataset idx and partition into the json
                        json["dataset_name"] = self.name
                        self.jsons_and_keys.append((json, key))
                        # if the dataset has script annotations, do not add identical data
                        if len(set([str(j["ann"]["instr"]) for j in task_jsons])) == 1:
                            break

        # return the true length of the loaded data
        return len(self.jsons_and_keys) if jsons else None

    def load_frames(self, key):
        """
        load image features from the disk
        """
        if not hasattr(self, "feats_lmdb"):
            self.feats_lmdb, self.feats = self.load_lmdb(self.feats_lmdb_path)
        feats_bytes = self.feats.get(key)
        feats_numpy = np.frombuffer(feats_bytes, dtype=np.float32).reshape(self.dataset_info["feat_shape"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frames = torch.tensor(feats_numpy)
        return frames

    def load_lmdb(self, lmdb_path):
        """
        load lmdb (should be executed in each worker on demand)
        """
        database = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=self.args.lmdb_max_readers,
        )
        cursor = database.begin(write=False)
        return database, cursor

    def __len__(self):
        """
        return dataset length
        """
        return self._length

    def __getitem__(self, idx):
        """
        get item at index idx
        """
        raise NotImplementedError

    @property
    def id(self):
        return self.partition + ":" + self.name + ";" + self.ann_type

    def __del__(self):
        """
        close the dataset
        """
        if hasattr(self, "feats_lmdb"):
            self.feats_lmdb.close()
        if hasattr(self, "masks_lmdb"):
            self.masks_lmdb.close()

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self.id)
