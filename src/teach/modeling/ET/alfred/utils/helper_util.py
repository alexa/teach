import torch
from vocab import Vocab as VocabBase


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class DataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class VocabWithLock(VocabBase):
    """vocab.Vocab with a lock for parallel computations."""

    def __init__(self, words=(), lock=None):
        self.lock = lock
        super().__init__(words)

    def word2index(self, word, train=False):
        """Original function copy with the self.lock call."""
        if isinstance(word, (list, tuple)):
            return [self.word2index(w, train=train) for w in word]
        with self.lock:
            self.counts[word] += train
            if word in self._word2index:
                return self._word2index[word]
            else:
                if train:
                    self._index2word += [word]
                    self._word2index[word] = len(self._word2index)
                else:
                    return self._handle_oov_word(word)
            index = self._word2index[word]
        return index


def identity(x):
    """
    pickable equivalent of lambda x: x
    """
    return x
