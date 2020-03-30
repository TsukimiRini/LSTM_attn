'''
@Author: tsukimi
@Date: 2020-03-29 22:59:27
@LastEditTime: 2020-03-30 13:16:34
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \\LSTM_attn\\data\\preprocess.py
'''
import torch
from tqdm import tqdm


class Dictionary(object):
    def __init__(self, idx=0):
        super().__init__()
        self.dic = {}
        self.idx = idx

    def _add(self, word):
        if self.dic.get(word) is None:
            self.dic[word] = self.idx
            self.idx += 1

    def translate(self, sents):
        _sents = []
        for sent in sents:
            _sent = []
            for word in sent:
                if self.dic.get(word) is None:
                    word = self.dic['<unk>']
                else:
                    word = self.dic[word]
                _sent += [word]
            _sents += [_sent]
        return _sents


class Vocab(Dictionary):
    def __init__(self):
        super().__init__(idx=2)
        self.dic = {'<pad>': 0, '<unk>': 1}

    def __call__(self, sents):
        for sent in sents:
            for word in sent:
                self._add(word)

    def translate(self, sents):
        return super().translate(sents)


class Labels(Dictionary):
    def __init__(self):
        super().__init__()

    def __call__(self, labels):
        for label in labels:
            self._add(label)

    def translate(self, labels):
        _labels = []
        for label in labels:
            label = self.dic[label]
            _labels += [label]
        return _labels


class DataProcess(object):
    def __init__(self, save_path="./corpus.pt", max_len=16):
        self.save_path = save_path
        self.max_len = max_len
        self.train_sents = []
        self.train_labels = []
        self.valid_sents = []
        self.valid_labels = []

        self.word_vocab = Vocab()
        self.label_lib = Labels()

    def parse_data(self, file_path, is_train=True):
        """Parse dataset to get vocab and label library.
        ---------------------------
        Arguments:
            file_path {String} -- file path of dataset
        ---------------------------
        Keyword Arguments:
            is_train {bool} -- whether the dataset is a train dataset (default: {True})
        """
        labels = []
        sents = []
        with open(file_path, 'r') as f:
            for line in tqdm(f):
                label, _, words = line.replace('\xf0', ' ').partition(' ')
                label = label.split(":")[0]
                words = words.strip().split()
                if len(words) > self.max_len:
                    words = words[:self.max_len]
                labels += [label]
                sents += [words]
        if is_train is True:
            self.word_vocab(sents)
            self.label_lib(labels)

            self.train_sents = sents
            self.train_labels = labels
        else:
            self.valid_sents = sents
            self.valid_labels = labels

    def process(self):
        self.parse_data("./train", is_train=True)
        self.parse_data("./valid", is_train=False)

        data = {
            "max_len": self.max_len,
            "train_sents": self.word_vocab.translate(self.train_sents),
            "train_labels": self.label_lib.translate(self.train_labels),
            "valid_sents": self.word_vocab.translate(self.valid_sents),
            "valid_labels": self.label_lib.translate(self.valid_labels),
            "dict": {
                "vocab": self.word_vocab.dic,
                "vacab_size": len(self.word_vocab.dic),
                "labels": self.label_lib.dic,
                "labels_size": len(self.label_lib.dic)
            }
        }

        torch.save(data, self.save_path)
        print("dataset preprocessing complted.")
        print("===============================")
        print("{} examples in train dataset.".format(len(self.train_labels)))
        print("{} examples in valid dataset.".format(len(self.valid_labels)))
        print("the size of vocabulary is: {}".format(len(self.word_vocab.dic)))
        print("the label library's size: {}".format(len(self.label_lib.dic)))


if __name__ == "__main__":
    prepro = DataProcess()
    prepro.process()
