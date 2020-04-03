'''
@Author: your name
@Date: 2020-04-01 12:45:05
@LastEditTime: 2020-04-02 11:45:17
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \\LSTM_attn\\model.py
'''
import torch
import torch.nn as nn


class LSTM_attn(nn.Module):
    def __init__(self,
                 label_size,
                 hidden,
                 bidirectional,
                 vocab_size,
                 emb_dim,
                 drop_out=0.1,
                 layers=1):
        super().__init__()
        self.label_size = label_size
        self.hidden_size = hidden
        self.num_layers = layers
        self.bidirectional = bidirectional
        self.tanh = nn.Tanh()
        self.emb = emb_dim
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden,
                            num_layers=layers,
                            dropout=drop_out,
                            bidirectional=bidirectional).cuda()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_dim,
                                      padding_idx=1).cuda()
        self.label = nn.Linear(self.hidden_size, self.label_size, bias=True)
        self.u = nn.Linear(16, self.hidden_size)
        if self.bidirectional:
            self.w = nn.Linear(self.hidden_size * 2,
                               self.hidden_size * 2)
        else:
            self.w = nn.Linear(self.hidden_size, self.hidden_size)

    def attn(self, lstm_output):
        """[summary]
        Arguments:
            lstm_output  -- [seq_len,bs,num_direction*hidden]
        """
        _lstm_output = lstm_output.permute(1, 0, 2)
        M = self.tanh(_lstm_output)
        alpha = nn.Softmax(dim=-1)(self.w(M))  # [bs, seq_len, _hidden]
        r = torch.bmm(_lstm_output, torch.transpose(alpha, 1, 2))
        r = self.tanh(r)
        r = self.u(torch.sum(r, -1))
        return r

    def forward(self, input):
        """main logic
        Arguments:
            input  -- [batch, max_len]
        """
        input = self.embedding(input)  # [batch, max_len, emb_dim]
        input = input.permute(1, 0, 2)  # [max_len, batch, emb_dim]
        assert input.size() == (16, 64, self.emb)
        max_len, bs, emb_dim = input.size()

        if self.bidirectional:
            h_0 = torch.zeros(self.num_layers * 2, bs, self.hidden_size).cuda()
            c_0 = torch.zeros(self.num_layers * 2, bs, self.hidden_size).cuda()
        else:
            h_0 = torch.zeros(self.num_layers, bs, self.hidden_size).cuda()
            c_0 = torch.zeros(self.num_layers, bs, self.hidden_size).cuda()

        lstm_output, (h_n, c_n) = self.lstm(input, (h_0, c_0))
        attn_output = self.attn(lstm_output)
        output = self.label(attn_output)

        return output
