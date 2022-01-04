from alfred.utils import data_util
from torch import nn


class Model(nn.Module):
    def __init__(self, args, embs_ann, vocab_out, pad, seg, for_inference=False):
        """
        Abstract model
        """
        nn.Module.__init__(self)
        self.args = args
        self.vocab_out = vocab_out
        self.pad, self.seg = pad, seg
        if for_inference:
            model_dir = args["model_dir"]
            dataset_info = data_util.read_dataset_info_for_inference(model_dir)
        else:
            dataset_info = data_util.read_dataset_info(args.data["train"][0])
        self.visual_tensor_shape = dataset_info["feat_shape"][1:]

        # create language and action embeddings
        self.embs_ann = nn.ModuleDict({})
        for emb_name, emb_size in embs_ann.items():
            self.embs_ann[emb_name] = nn.Embedding(emb_size, args.demb)

        # dropouts
        self.dropout_vis = nn.Dropout(args.dropout["vis"], inplace=True)
        self.dropout_lang = nn.Dropout2d(args.dropout["lang"])

    def init_weights(self, init_range=0.1):
        """
        init linear layers in embeddings
        """
        for emb_ann in self.embs_ann.values():
            emb_ann.weight.data.uniform_(-init_range, init_range)

    def compute_metrics(self, model_out, gt_dict, metrics_dict, verbose):
        """
        compute model-specific metrics and put it to metrics dict
        """
        raise NotImplementedError

    def forward(self, vocab, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        raise NotImplementedError()

    def compute_batch_loss(self, model_out, gt_dict):
        """
        compute the loss function for a single batch
        """
        raise NotImplementedError()

    def compute_loss(self, model_outs, gt_dicts):
        """
        compute the loss function for several batches
        """
        # compute losses for each batch
        losses = {}
        for dataset_key in model_outs.keys():
            losses[dataset_key] = self.compute_batch_loss(model_outs[dataset_key], gt_dicts[dataset_key])
        return losses
