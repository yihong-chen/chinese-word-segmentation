import torch
import torch.nn.functional as F
from torch import nn, autograd


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, use_cuda):
        super(LSTMTagger, self).__init__()
        self.use_cuda = use_cuda
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hid = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        cel = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        if self.use_cuda:
            hid = hid.cuda()
            cel = cel.cuda()
        return (hid, cel)

    def forward(self, sentence):
        try:
            embeds = self.word_embeddings(sentence)
        except:
            print(sentence)
            exit(1)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores