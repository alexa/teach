from torch import nn
from torch.nn import functional as F


class SelfAttn(nn.Module):
    """
    self-attention with learnable parameters
    """

    def __init__(self, dhid):
        super().__init__()
        self.scorer = nn.Linear(dhid, 1)
        # scorer: dhid x 1

    def forward(self, inp):
        # inp: batch_size x seq_len x dhid
        scores = F.softmax(self.scorer(inp), dim=1)
        # scores: batch_size x seq_len x 1
        cont = scores.transpose(1, 2).bmm(inp).squeeze(1)
        # cont: batch_size x seq_len
        return cont


class DotAttn(nn.Module):
    """
    dot-attention (or soft-attention)
    """

    def forward(self, inp, h):
        # inp: batch_size x seq_len x dhid
        # h: batch_size x dhid
        score = self.softmax(inp, h)
        # score: batch_size x seq_len x 1
        score_expanded = score.expand_as(inp)
        # score_expanded: batch_size x seq_len x dhid
        # output: batch_size x dhid
        return score_expanded.mul(inp).sum(1), score

    def softmax(self, inp, h):
        raw_score = inp.bmm(h.unsqueeze(2))
        # raw_score: batch_size x seq_len x 1
        score = F.softmax(raw_score, dim=1)
        # score: batch_size x seq_len x 1
        return score
