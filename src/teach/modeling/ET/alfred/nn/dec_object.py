import os

import torch
from alfred import constants
from torch import nn


class ObjectClassifier(nn.Module):
    """
    object classifier module (a single FF layer)
    """

    def __init__(self, input_size):
        super().__init__()
        vocab_obj_path = os.path.join(constants.ET_ROOT, constants.OBJ_CLS_VOCAB)
        vocab_obj = torch.load(vocab_obj_path)
        num_classes = len(vocab_obj)
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out
