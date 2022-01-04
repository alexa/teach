import json
import logging
import os
import pickle
import re
import shutil
import string
from copy import deepcopy

import lmdb
import torch
from alfred import constants
from alfred.utils import helper_util
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from teach.logger import create_logger

logger = create_logger(__name__, level=logging.INFO)


def read_images(image_path_list):
    images = []
    for image_path in image_path_list:
        image_orig = Image.open(image_path)
        images.append(image_orig.copy())
        image_orig.close()
    return images


def read_traj_images(json_path, image_folder):
    with open(json_path) as json_file:
        json_dict = json.load(json_file)

    images_dir = json_path.parents[2] / image_folder / json_path.parts[-2] / json_path.parts[-1].split(".")[0]

    fimages = [images_dir / im for im in json_dict["driver_image_history"] + json_dict["driver_images_future"]]
    logger.debug("Loading images from %s" % images_dir)
    logger.debug("Expected image files: %s" % "\n\t".join([str(x) for x in fimages]))

    if not all([os.path.exists(path) for path in fimages]):
        return None
    assert len(fimages) > 0
    images = read_images(fimages)
    return images


def extract_features(images, extractor):
    if images is None:
        return None
    feat = extractor.featurize(images, batch=8)
    return feat.cpu()


def process_traj(traj_orig, traj_path, r_idx, preprocessor):
    # copy trajectory
    traj = traj_orig.copy()
    # root & split
    traj["root"] = str(traj_path)
    partition = traj_path.parts[-2]
    traj["split"] = partition
    traj["repeat_idx"] = r_idx
    # numericalize actions for train/valid splits
    preprocessor.process_actions(traj_orig, traj)
    # numericalize language
    if "test" in partition:
        preprocessor.process_language(traj_orig, traj, r_idx, is_test_split=True)
    else:
        preprocessor.process_language(traj_orig, traj, r_idx, is_test_split=False)
    return traj


def gather_feats(files, output_path):
    if output_path.is_dir():
        shutil.rmtree(output_path)
    lmdb_feats = lmdb.open(str(output_path), 700 * 1024 ** 3, writemap=True)
    with lmdb_feats.begin(write=True) as txn_feats:
        for idx, path in tqdm(enumerate(files)):
            traj_feats = torch.load(path).numpy()
            txn_feats.put("{:06}".format(idx).encode("ascii"), traj_feats.tobytes())
    lmdb_feats.close()


def gather_jsons(files, output_path):
    if output_path.exists():
        os.remove(output_path)
    jsons = {}
    for idx, path in tqdm(enumerate(files)):
        with open(path, "rb") as f:
            jsons_idx = pickle.load(f)
            jsons["{:06}".format(idx).encode("ascii")] = jsons_idx
    with output_path.open("wb") as f:
        pickle.dump(jsons, f)


def get_preprocessor(PreprocessorClass, subgoal_ann, lock, vocab_path=None, task_type="edh"):
    if vocab_path is None:
        init_words = ["<<pad>>", "<<seg>>", "<<goal>>", "<<mask>>"]
    else:
        init_words = []
    vocabs_with_lock = {
        "word": helper_util.VocabWithLock(deepcopy(init_words), lock),
        "action_low": helper_util.VocabWithLock(deepcopy(init_words), lock),
        "action_high": helper_util.VocabWithLock(deepcopy(init_words), lock),
    }
    if vocab_path is not None:
        vocabs_loaded = torch.load(vocab_path)
        for vocab_name, vocab in vocabs_with_lock.items():
            loaded_dict = vocabs_loaded[vocab_name].to_dict()
            for _i, w in enumerate(loaded_dict["index2word"]):
                vocab.word2index(w, train=True)
                vocab.counts[w] = loaded_dict["counts"][w]

    actions_high_init_words = [
        "Navigate",
        "Pickup",
        "Place",
        "Open",
        "Close",
        "ToggleOn",
        "ToggleOff",
        "Slice",
        "Pour",
        "object",
    ]

    # Reset low actions vocab to empty because Simbot vocab is different
    actions_low_init_words = [
        "Stop",
        "Forward",
        "Backward",
        "Turn Left",
        "Turn Right",
        "Look Up",
        "Look Down",
        "Pan Left",
        "Pan Right",
        "Navigation",
        "Pickup",
        "Place",
        "Open",
        "Close",
        "ToggleOn",
        "ToggleOff",
        "Slice",
        "Pour",
    ]
    if task_type == "tfd":
        actions_low_init_words.append("Text")

    vocabs_with_lock["action_low"] = helper_util.VocabWithLock(actions_low_init_words, lock)
    vocabs_with_lock["action_high"] = helper_util.VocabWithLock(actions_high_init_words, lock)
    vocab_obj = torch.load(os.path.join(constants.ET_ROOT, constants.OBJ_CLS_VOCAB)).to_dict()
    logger.debug("In get_preprocessor, vocab_obj = %s" % vocab_obj["index2word"])
    for _i, w in enumerate(vocab_obj["index2word"]):
        vocabs_with_lock["action_high"].word2index(w, train=True)
        vocabs_with_lock["action_high"].counts[w] = vocab_obj["counts"][w]

    preprocessor = PreprocessorClass(vocabs_with_lock, subgoal_ann)
    return preprocessor


def tensorize_and_pad(batch, device, pad):
    """
    cast values to torch tensors, put them to the correct device and pad sequences
    """
    device = torch.device(device)
    input_dict, gt_dict, feat_dict = dict(), dict(), dict()
    traj_data, feat_list = list(zip(*batch))
    for key in feat_list[0].keys():
        feat_dict[key] = [el[key] for el in feat_list]
    # feat_dict keys that start with these substrings will be assigned to input_dict
    input_keys = {"lang", "frames"}
    # the rest of the keys will be assigned to gt_dict

    for k, v in feat_dict.items():
        dict_assign = input_dict if any([k.startswith(s) for s in input_keys]) else gt_dict
        if k.startswith("lang"):
            # no preprocessing should be done here
            seqs = [torch.tensor(vv if vv is not None else [pad, pad], device=device).long() for vv in v]
            pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
            dict_assign[k] = pad_seq
            dict_assign["lengths_" + k] = torch.tensor(list(map(len, seqs)))
            length_max_key = "length_" + k + "_max"
            if ":" in k:
                # for translated length keys (e.g. lang:lmdb/1x_det) we should use different names
                length_max_key = "length_" + k.split(":")[0] + "_max:" + ":".join(k.split(":")[1:])
            dict_assign[length_max_key] = max(map(len, seqs))
        elif k in {"object"}:
            # convert lists with object indices to tensors
            seqs = [torch.tensor(vv, device=device, dtype=torch.long) for vv in v if len(vv) > 0]
            dict_assign[k] = seqs
        elif k in {"frames"}:
            # frames features were loaded from the disk as tensors
            seqs = [vv.clone().detach().to(device).type(torch.float) for vv in v]
            pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
            dict_assign[k] = pad_seq
            dict_assign["lengths_" + k] = torch.tensor(list(map(len, seqs)))
            dict_assign["length_" + k + "_max"] = max(map(len, seqs))
        else:
            # default: tensorize and pad sequence
            seqs = [torch.tensor(vv, device=device, dtype=torch.long) for vv in v]
            pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
            dict_assign[k] = pad_seq
    return traj_data, input_dict, gt_dict


def sample_batches(iterators, device, pad, args):
    """
    sample a batch from each iterator, return Nones if the iterator is empty
    """
    batches_dict = {}
    for dataset_id, iterator in iterators.items():
        try:
            batches = next(iterator)
        except StopIteration as e:
            return None
        dataset_name = dataset_id.split(":")[1]
        traj_data, input_dict, gt_dict = tensorize_and_pad(batches, device, pad)
        batches_dict[dataset_name] = (traj_data, input_dict, gt_dict)
    return batches_dict


def load_vocab(name, ann_type="lang"):
    """
    load a vocabulary from the dataset
    """
    path = os.path.join(constants.ET_DATA, name, constants.VOCAB_FILENAME)
    logger.info("In load_vocab, loading vocab from %s" % path)
    vocab_dict = torch.load(path)
    # set name and annotation types
    for vocab in vocab_dict.values():
        vocab.name = name
        vocab.ann_type = ann_type
    return vocab_dict


def load_vocab_for_inference(model_dir, name, ann_type="lang"):
    path = os.path.join(model_dir, constants.VOCAB_FILENAME)
    logger.info("In load_vocab, loading vocab from %s" % path)
    vocab_dict = torch.load(path)
    # set name and annotation types
    for vocab in vocab_dict.values():
        vocab.name = name
        vocab.ann_type = ann_type
    return vocab_dict


def get_feat_shape(visual_archi, compress_type=None):
    """
    Get feat shape depending on the training archi and compress type
    """
    if visual_archi == "fasterrcnn":
        # the RCNN model should be trained with min_size=224
        feat_shape = (-1, 2048, 7, 7)
    elif visual_archi == "maskrcnn":
        # the RCNN model should be trained with min_size=800
        feat_shape = (-1, 2048, 10, 10)
    elif visual_archi == "resnet18":
        feat_shape = (-1, 512, 7, 7)
    else:
        raise NotImplementedError("Unknown archi {}".format(visual_archi))

    if compress_type is not None:
        if not re.match(r"\d+x", compress_type):
            raise NotImplementedError("Unknown compress type {}".format(compress_type))
        compress_times = int(compress_type[:-1])
        feat_shape = (
            feat_shape[0],
            feat_shape[1] // compress_times,
            feat_shape[2],
            feat_shape[3],
        )
    return feat_shape


def feat_compress(feat, compress_type):
    """
    Compress features by channel average pooling
    """
    assert re.match(r"\d+x", compress_type) and len(feat.shape) == 4
    times = int(compress_type[:-1])
    assert feat.shape[1] % times == 0
    feat = feat.reshape((feat.shape[0], times, feat.shape[1] // times, feat.shape[2], feat.shape[3]))
    feat = feat.mean(dim=1)
    return feat


def read_dataset_info(data_name):
    """
    Read dataset a feature shape and a feature extractor checkpoint path
    """
    path = os.path.join(constants.ET_DATA, data_name, "params.json")
    with open(path, "r") as f_params:
        params = json.load(f_params)
    return params


def read_dataset_info_for_inference(model_dir):
    """
    Read dataset a feature shape and a feature extractor checkpoint path from file stored in model checkpoint
    """
    path = os.path.join(model_dir, "params.json")
    logger.info("Reading dataset info from %s for model dir %s" % (path, model_dir))
    with open(path, "r") as f_params:
        params = json.load(f_params)
    return params


def remove_spaces(s):
    cs = " ".join(s.split())
    return cs


def remove_spaces_and_lower(s):
    cs = remove_spaces(s)
    cs = cs.lower()
    return cs


def remove_punctuation(s):
    cs = s.translate(str.maketrans("", "", string.punctuation))
    cs = remove_spaces_and_lower(cs)
    return cs
