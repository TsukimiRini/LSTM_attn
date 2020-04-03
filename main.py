'''
@Author: your name
@Date: 2020-03-30 20:58:29
@LastEditTime: 2020-04-02 11:52:57
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: \\LSTM_attn\\main.py
'''
import torch
from data_loader import DataLoader
from model import LSTM_attn
from _model import bilstm_attn
from tqdm import tqdm
import time

data_path = "./data/corpus.pt"
model_path = "./model/"
emb_size = 64
hidden_size = 32
lr = 0.001
Epoch = 1000
batch_size = 64
train_from = "./model/best_model"


def load_data():
    data = torch.load(data_path)
    vocab_size = data["dict"]["vacab_size"]
    label_size = data["dict"]["labels_size"]
    train_sents = data["train_sents"]
    train_labels = data["train_labels"]
    valid_sents = data["valid_sents"]
    valid_labels = data["valid_labels"]
    train_len = len(train_sents)
    valid_len = len(valid_sents)
    train_data = DataLoader(train_sents, train_labels)
    valid_data = DataLoader(valid_sents, valid_labels, shuffle=False)

    return train_sents, train_labels, vocab_size, label_size, train_data, valid_data, train_len, valid_len


def train(train_data):
    losses = []
    sum = 0
    for sents, label in tqdm(train_data, desc="training"):
        optimizer.zero_grad()
        output = model(sents)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        losses += [loss.data]
        sum += loss.data
    return sum / train_len


def validate(valid_data):
    losses = []
    corrects = sum = 0
    for sents, label in tqdm(valid_data,
                             desc="validating",
                             mininterval=1,
                             leave=False):
        output = model(sents)
        loss = criterion(output, label.long())
        losses += [loss.data]
        sum += loss.data
        output_label = torch.argmax(output, dim=1)
        corrects += (output_label.data == label.data).sum()
    return sum / valid_len, corrects, corrects * 100.0 / valid_len


start = time.time()
train_sents, train_labels, vocab_size, label_size, train_data, valid_data, train_len, valid_len = load_data(
)
model = LSTM_attn(label_size=label_size,
                  hidden=hidden_size,
                  bidirectional=True,
                  vocab_size=vocab_size,
                  emb_dim=emb_size,
                  drop_out=0.1).cuda()
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
criterion = torch.nn.CrossEntropyLoss()
bestacc = -1
t = 1
if train_from:
    checkpoint = torch.load(train_from)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    bestacc = checkpoint["acc"]
    t = checkpoint["epoch"]
for epoch in range(t, Epoch + 1):
    avg_loss = train(train_data)
    print('-' * 10)
    print("Epoch {} train avg loss: {}".format(epoch, avg_loss))
    avg_loss, _, acc = validate(valid_data)
    print("Epoch {} valid avg loss: {}, acc: {}".format(epoch, avg_loss, acc))
    print("Epoch {} consumed {}s".format(epoch, time.time() - start))
    train_data = DataLoader(train_sents, train_labels)
    if bestacc < acc:
        acc = bestacc
        model_state_dict = model.state_dict()
        model_source = {"model": model_state_dict,
        "optimizer": optimizer.state_dict(),
        "acc": acc,
        "epoch": epoch}
        torch.save(model_source, model_path + "best_model")
    if epoch % 100 == 0:
        model_state_dict = model.state_dict()
        model_source = {"model": model_state_dict,
        "optimizer": optimizer.state_dict(),
        "acc": acc,
        "epoch": epoch}
        torch.save(model_source, model_path + "model_" + str(epoch))
print('=' * 10)
print("training completed")
