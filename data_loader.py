'''
@Author: your name
@Date: 2020-04-01 14:59:22
@LastEditTime: 2020-04-02 11:15:17
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \\LSTM_attn\\data_loader.py
'''
import numpy as np
import torch


class DataLoader(object):
    def __init__(self, src, label, batch_size=64, shuffle=True):
        super().__init__()
        self.src = np.asarray(src)
        self.src_size = len(src)
        self.stop_step = self.src_size // batch_size
        self.label = np.asarray(label)
        self.bs = batch_size
        self.max_len = 16
        if shuffle:
            self.shuffle()
        self.idx = 0
        self.step = 0

    def shuffle(self):
        indices = np.arange(0, self.src_size)
        indices = np.random.shuffle(indices)
        self.src = self.src[indices]
        self.label = self.label[indices]
        self.src = self.src.tolist()[0]
        self.label = self.label.tolist()[0]

    def __iter__(self):
        return self

    def __next__(self):
        if self.step == self.stop_step:
            self.step = 0
            self.idx = 0
            raise StopIteration()
        minisrc = self.src[self.idx:self.idx + self.bs]
        minisrc = [sent + [0]*(self.max_len-len(sent)) for sent in minisrc]
        minisrc = torch.tensor(minisrc).cuda()
        minilabel = torch.tensor(self.label[self.idx:self.idx +
                                            self.bs]).cuda()
        self.idx += self.bs
        self.step += 1
        return minisrc, minilabel
