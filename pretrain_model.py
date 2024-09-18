import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CrossEncoder(nn.Module):
    def __init__(self):
        super(CrossEncoder, self).__init__()
        self.dataset = 'MER2024'
        if self.dataset == 'MER2024':
            self.audio_dim = 1024
            self.text_dim = 5120
            self.video_dim = 768
            self.output_dim = 6

        self.dropout = 0.5  # 0
        self.hidden_dim = 256
        self.num_heads = 8
        self.layers = 2
        self.cls_layers = '256, 128'

        self.device = 'cuda:1'
        # model
        self.netA = nn.TransformerEncoder(nn.TransformerEncoderLayer
                                          (d_model=self.audio_dim,
                                           nhead=self.num_heads,
                                           # nhead=5,
                                           dim_feedforward=2048,
                                           dropout=self.dropout, batch_first=True),
                                          num_layers=self.layers)
        self.netV = nn.TransformerEncoder(nn.TransformerEncoderLayer
                                          (d_model=self.video_dim,
                                           nhead=self.num_heads,
                                           # nhead=6,
                                           dim_feedforward=2048,
                                           dropout=self.dropout, batch_first=True),
                                          num_layers=self.layers)
        self.netL = nn.TransformerEncoder(nn.TransformerEncoderLayer
                                          (d_model=self.text_dim,
                                           nhead=self.num_heads,
                                           # nhead=8,
                                           dim_feedforward=2048,
                                           dropout=self.dropout, batch_first=True),
                                          num_layers=self.layers)
        self.netlinearA = nn.Linear(self.audio_dim, self.hidden_dim)
        self.netlinearV = nn.Linear(self.video_dim, self.hidden_dim)
        self.netlinearT = nn.Linear(self.text_dim, self.hidden_dim)


        self.criterion = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()
        cls_layers = list(map(lambda x: int(x), self.cls_layers.split(',')))

        # self.netC = FcClassifier(self.hidden_dim * 3, cls_layers, output_dim=6, dropout=self.dropout,
        #                          use_bn=True)
        # self.netC1 = FcClassifier(self.hidden_dim * 3, cls_layers, output_dim=0, dropout=self.dropout,
        #                          use_bn=True)
        self.netC2 = FcClassifier(self.hidden_dim, cls_layers, output_dim=self.output_dim1, dropout=self.dropout,
                                 use_bn=True)
        self.netC3 = FcClassifier(self.hidden_dim, cls_layers, output_dim=self.output_dim1, dropout=self.dropout,
                                 use_bn=True)
        self.netC1 = FcClassifier(self.hidden_dim * 3, cls_layers, output_dim=0, dropout=self.dropout,
                                  use_bn=True)

        self.cls_token_A = nn.Parameter(torch.zeros(1, 1, self.audio_dim), requires_grad=True).to(self.device)
        self.cls_token_V = nn.Parameter(torch.zeros(1, 1, self.video_dim), requires_grad=True).to(self.device)
        self.cls_token_L = nn.Parameter(torch.zeros(1, 1, self.text_dim), requires_grad=True).to(self.device)

        self.positional_encoding_A = nn.Parameter(torch.zeros(1, 2, self.audio_dim)).to(self.device)
        self.positional_encoding_V = nn.Parameter(torch.zeros(1, 2, self.video_dim)).to(self.device)
        self.positional_encoding_L = nn.Parameter(torch.zeros(1, 2, self.text_dim)).to(self.device)



    def forward(self, a, v, l):

        feat_fusion = torch.zeros(10, 256, device=self.device)
        if a is not None:
            b = a.size(0)
            self.cls_token_A1 = self.cls_token_A.expand(b, -1, -1)
            feat_A = torch.cat((self.cls_token_A1, a.unsqueeze(1)), dim=1)
            feat_A = feat_A + self.positional_encoding_A[:, :feat_A.size(1), :]
            feat_A = self.netA(feat_A)
            feat_A = self.netlinearA(feat_A)

            cls_output_A = feat_A[:, 0, :]
            cls_output_A1, _ = self.netC1(cls_output_A)
            cls_output_A1 = F.softmax(cls_output_A1, dim=1)
            cls_output_A1 = torch.argmax(cls_output_A1)

        if v is not None:
            b = v.size(0)
            self.cls_token_V1 = self.cls_token_V.expand(b, -1, -1)
            feat_V = torch.cat((self.cls_token_V1, v.unsqueeze(1)), dim=1)
            feat_V = feat_V + self.positional_encoding_V[:, :feat_V.size(1), :]
            feat_V = self.netV(feat_V)
            feat_V = self.netlinearV(feat_V)

            cls_output_V = feat_V[:, 0, :]
            cls_output_V1, _ = self.netC2(cls_output_V)


        if l is not None:
            b = l.size(0)
            self.cls_token_L1 = self.cls_token_L.expand(b, -1, -1)
            feat_L = torch.cat((self.cls_token_L1, l.unsqueeze(1)), dim=1)
            feat_L = feat_L + self.positional_encoding_L[:, :feat_L.size(1), :]
            feat_L = self.netL(feat_L)
            feat_L = self.netlinearT(feat_L)

            cls_output_L = feat_L[:, 0, :]
            cls_output_L1, _ = self.netC3(cls_output_L)
