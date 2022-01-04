import logging

import numpy as np
import torch
from alfred.model import base
from alfred.nn.enc_lang import EncoderLang
from alfred.nn.enc_visual import FeatureFlat
from alfred.nn.enc_vl import EncoderVL
from alfred.nn.encodings import PosLangEncoding
from alfred.utils import model_util
from torch import nn
from torch.nn import functional as F

from teach.logger import create_logger

logger = create_logger(__name__, level=logging.INFO)


class Model(base.Model):
    def __init__(self, args, embs_ann, vocab_out, pad, seg, for_inference=False):
        """
        speaker model
        """
        super().__init__(args, embs_ann, vocab_out, pad, seg, for_inference)

        # encoder and visual embeddings
        self.encoder_vl, self.encoder_lang = None, None
        if any("frames" in ann_type for ann_type in args.data["ann_type"]):
            # create a multi-modal encoder
            self.encoder_vl = EncoderVL(args)
            # create feature embeddings
            self.vis_feat = FeatureFlat(input_shape=self.visual_tensor_shape, output_size=args.demb)
        else:
            # create an encoder for language only
            self.encoder_lang = EncoderLang(args.encoder_layers, args, embs_ann)

        # decoder parts
        decoder_layer = nn.TransformerDecoderLayer(
            args.demb,
            args.decoder_lang["heads"],
            args.decoder_lang["demb"],
            args.decoder_lang["dropout"],
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, args.decoder_lang["layers"])
        self.enc_pos = PosLangEncoding(args.demb) if args.decoder_lang["pos_enc"] else None
        self.emb_subgoal = nn.Embedding(len(vocab_out), args.demb)

        # final touch
        self.init_weights()

    def encode_vl(self, vocab, **inputs):
        """
        apply the VL encoder to the inputs
        """
        lang = inputs["lang"] if "lang" in inputs else None
        frames = inputs["frames"] if "frames" in inputs else None
        device = lang.device if lang is not None else frames.device
        assert inputs is not None or frames is not None
        batch_size = len(lang if lang is not None else frames)
        # embed language if the model should see them
        if lang is not None:
            emb_lang = self.embed_lang(lang, self.embs_ann[vocab.name])
            lengths_lang = inputs["lengths_lang"]
        else:
            emb_lang = torch.zeros([batch_size, 0, self.args.demb]).to(device)
            lengths_lang = torch.tensor([0] * batch_size)

        # embed frames if the model should see them
        if frames is not None:
            emb_frames = self.embed_frames(frames)
            lengths_frames = inputs["lengths_frames"]
            length_frames_max = inputs["length_frames_max"]
        else:
            emb_frames = torch.zeros([batch_size, 0, self.args.demb]).to(device)
            lengths_frames, length_frames_max = torch.tensor([0] * batch_size), 0
        # speaker does not use the actions
        emb_actions = torch.zeros([batch_size, 0, self.args.demb]).to(device)
        lengths_actions = torch.tensor([0] * batch_size)
        # encode inputs
        hiddens, hiddens_padding = self.encoder_vl(
            emb_lang,
            emb_frames,
            emb_actions,
            lengths_lang,
            lengths_frames,
            lengths_actions,
            length_frames_max,
            attn_masks=False,
        )
        return hiddens, hiddens_padding

    def encode_lang(self, vocab, lang_pad):
        """
        apply the language encoder to the inputs
        """
        embedder_lang = self.embs_ann[vocab.name]
        emb_lang, lengths_lang = self.encoder_lang(lang_pad, embedder_lang, vocab, self.pad)
        emb_padding = torch.zeros(emb_lang.shape[:2], device=emb_lang.device).bool()
        for i, len_l in enumerate(lengths_lang):
            emb_padding[i, len_l:] = True
        return emb_lang, emb_padding

    def encode_inputs(self, vocab, **inputs):
        """
        apply the VL or language encoder to the inputs
        """
        if self.encoder_vl is not None:
            hiddens, hiddens_padding = self.encode_vl(vocab, **inputs)
        else:
            hiddens, hiddens_padding = self.encode_lang(vocab, inputs["lang"])
        return hiddens, hiddens_padding

    def forward(self, vocab, **inputs):
        """
        forward the model for multiple time-steps (used for training)
        """
        # pass inputs to the encoder
        hiddens, hiddens_padding = self.encode_inputs(vocab, **inputs)
        hiddens = self.enc_pos(hiddens) if self.enc_pos else hiddens
        # generate masks
        lang_target = inputs["action"]
        target_mask = model_util.triangular_mask(lang_target.size(1), lang_target.device)
        # right shift the targets
        lang_target = lang_target.clone().detach()
        lang_target = torch.roll(lang_target, 1, 1)
        lang_target[:, 0] = self.seg
        # embed targets and add position encodings
        target = self.embed_lang(lang_target, self.emb_subgoal)
        target = self.enc_pos(target) if self.enc_pos else target

        # decode the outputs with transformer
        decoder_out = self.decoder(
            tgt=target.transpose(0, 1),
            memory=hiddens.transpose(0, 1),
            # to avoid looking at the future tokens (the ones on the right)
            tgt_mask=target_mask,
            # avoid looking on padding of the src
            memory_key_padding_mask=hiddens_padding,
        ).transpose(0, 1)
        # apply a linear layer
        decoder_out_flat = decoder_out.reshape(-1, self.args.demb)
        lang_out_flat = decoder_out_flat.mm(self.emb_subgoal.weight.t())
        output = {"lang": lang_out_flat.view(len(decoder_out), -1, lang_out_flat.shape[-1])}
        return output

    def embed_frames(self, frames_pad):
        """
        take a list of frames tensors, pad it, apply dropout and extract embeddings
        """
        self.dropout_vis(frames_pad)
        frames_4d = frames_pad.view(-1, *frames_pad.shape[2:])
        frames_pad_emb = self.vis_feat(frames_4d).view(*frames_pad.shape[:2], -1)
        return frames_pad_emb

    def embed_lang(self, lang_pad, embedder):
        """
        embed goal+instr language
        """
        lang_pad_emb = embedder(lang_pad)
        lang_pad_emb = self.dropout_lang(lang_pad_emb)
        return lang_pad_emb

    def compute_batch_loss(self, model_out, gt_dict):
        """
        language translation loss function
        """
        p_lang = model_out["lang"].view(-1, model_out["lang"].shape[-1])
        l_lang = gt_dict["action"].view(-1)
        loss_lang = F.cross_entropy(p_lang, l_lang, reduction="none").mean()
        return {"lang": loss_lang}

    def init_weights(self, init_range=0.1):
        """
        init embeddings uniformly
        """
        super().init_weights(init_range)
        self.emb_subgoal.weight.data.uniform_(-init_range, init_range)

    def compute_metrics(self, model_out, gt_dict, metrics_dict, verbose=False):
        """
        compute exact matching and f1 score for action predictions
        """
        pred_tokens = model_out["lang"].max(2)[1].tolist()
        pred_lang = model_util.tokens_to_lang(pred_tokens, self.vocab_out, {self.pad}, join=False)
        gt_lang = model_util.tokens_to_lang(gt_dict["action"], self.vocab_out, {self.pad}, join=False)
        pred_lang_strs = [" ".join(s) for s in pred_lang]
        gt_lang_strs = [" ".join(s) for s in gt_lang]
        model_util.compute_f1_and_exact(metrics_dict, pred_lang_strs, gt_lang_strs, "lang")
        if verbose:
            logger.debug("Lang GT:\n{}".format(gt_lang_strs[0]))
            logger.debug("Lang predictions:\n{}".format(pred_lang_strs[0]))
            logger.debug("EM = {}, F1 = {}".format(metrics_dict["lang/exact"][-1], metrics_dict["lang/f1"][-1]))

    def translate(self, vocab_in, max_decode=300, num_pad_stop=3, **inputs):
        """
        lang and frames has shapes [1, LEN]
        """
        # prepare
        batch_size = len(inputs["lang"] if "lang" in inputs else inputs["frames"])
        device = (inputs["lang"] if "lang" in inputs else inputs["frames"]).device
        # pass inputs to the encoder
        hiddens, hiddens_padding = self.encode_inputs(vocab_in, **inputs)
        assert len(hiddens) == batch_size

        # start the decoding
        lang_cur = [[self.seg] for _ in range(batch_size)]
        for i in range(max_decode):
            tensor_cur = torch.tensor(lang_cur).to(device)
            emb_cur = self.embed_lang(tensor_cur, self.emb_subgoal)
            if self.enc_pos:
                emb_cur = self.enc_pos(emb_cur)
            mask_cur = model_util.triangular_mask(i + 1, device)

            decoder_out = self.decoder(
                tgt=emb_cur.transpose(0, 1),
                memory=hiddens.transpose(0, 1),
                tgt_mask=mask_cur,
                # avoid looking on padding of the src
                memory_key_padding_mask=hiddens_padding,
            ).transpose(0, 1)

            # apply a linear layer
            decoder_out_flat = decoder_out.reshape(-1, self.args.demb)
            lang_out_flat = decoder_out_flat.mm(self.emb_subgoal.weight.t())
            lang_out = lang_out_flat.view(batch_size, -1, lang_out_flat.shape[-1])
            tokens_out = lang_out.max(2)[1]
            for j in range(batch_size):
                lang_cur[j].append(tokens_out[i, -1].item())
            if len(tokens_out[0]) > num_pad_stop and (np.array(lang_cur)[:, -num_pad_stop:] == self.pad).all():
                break

        lang_result = [l[1:] for l in lang_cur]
        lang_result = [[t for t in tokens if t != self.pad] for tokens in lang_result]
        return lang_result
