from collections import Counter, OrderedDict
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab, vocab

Tensor = torch.Tensor


def split_and_flat(corpus: List[str]) -> List[str]:
    corpus_ = [x.split() for x in corpus]
    corpus = [x for y in corpus_ for x in y]
    return corpus


def build_vocab(corpus: List[str], oov: str = "<OOV>", pad: str = "<PAD>") -> Vocab:
    data = split_and_flat(corpus)
    counter = Counter(data)
    ordered_dict = OrderedDict(counter)
    v1 = vocab(ordered_dict, specials=[pad, oov])
    v1.set_default_index(v1[oov])
    return v1


def tokenize(corpus: List[str], v: Vocab) -> Tensor:
    batch = []
    for sent in corpus:
        batch.append(torch.tensor([v[word] for word in sent.split()]))
    return pad_sequence(batch, batch_first=True)
