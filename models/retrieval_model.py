import json
import os
from collections import OrderedDict
from models.base_model import BaseModel
from models.networks.classifier import FcClassifier
from models.utils.config import OptConfig
from models.utils.cmd import CMD
from models.retrieval_augmented import Retrieval_augment
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
from models.CrossEncoder import CrossEncoder


class MULTICONTRASTIVEModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'],
                            help='visual embedding method,last,mean or atten')
        parser.add_argument('--AE_layers', type=str, default='128,64,32',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--pretrained_path', type=str, help='where to load pretrained encoder network')
        parser.add_argument('--pretrained_invariant_path', type=str,
                            help='where to load pretrained invariant encoder network')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of mse loss')
        parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight of cycle loss')
        parser.add_argument('--invariant_weight', type=float, default=1.0, help='weight of invariant loss')
        parser.add_argument('--share_weight', action='store_true',
                            help='share weight of forward and backward autoencoders')
        parser.add_argument('--image_dir', type=str, default='./invariant_image', help='models image are saved here')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.args = opt
        # hyper parameters
        self.batch_size = opt.batch_size

        self.dataset = 'MER2024'
        if self.dataset == 'MER2024':
            self.audio_dim = 1024
            self.text_dim = 5120
            self.video_dim = 768
            self.output_dim = 6
        self.dropout = 0.5
        self.hidden_dim = opt.embd_size_a
        self.num_heads = 8
        self.layers = 1
        self.cls_layers = '256, 128'

        self.cls_layers = list(map(lambda x: int(x), self.cls_layers.split(',')))

        self.netlinearA = nn.Linear(self.audio_dim, self.hidden_dim)
        self.netlinearV = nn.Linear(self.video_dim, self.hidden_dim)
        self.netlinearT = nn.Linear(self.text_dim, self.hidden_dim)

        self.loss_names = ['CE']
        self.model_names = ['C', 'linearA', 'linearV', 'linearT']  # 六个模块的名称

        self.model_names.append('A')
        # # lexical model 
        self.model_names.append('L')
        # # visual model
        self.model_names.append('V')
        
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))

        self.cls_token_A = nn.Parameter(torch.zeros(1, 1, self.audio_dim), requires_grad=True).to(self.device)
        self.cls_token_V = nn.Parameter(torch.zeros(1, 1, self.video_dim), requires_grad=True).to(self.device)
        self.cls_token_L = nn.Parameter(torch.zeros(1, 1, self.text_dim), requires_grad=True).to(self.device)

        self.positional_encoding_A = nn.Parameter(torch.zeros(1, 2, self.audio_dim)).to(self.device)
        self.positional_encoding_V = nn.Parameter(torch.zeros(1, 2, self.video_dim)).to(self.device)
        self.positional_encoding_L = nn.Parameter(torch.zeros(1, 2, self.text_dim)).to(self.device)

        encoder_layer_A = TransformerEncoderLayer(d_model=self.audio_dim, nhead=8, dim_feedforward=self.hidden_dim,
                                                  batch_first=True)
        self.netA = TransformerEncoder(encoder_layer_A, num_layers=1)
        encoder_layer_V = TransformerEncoderLayer(d_model=self.video_dim, nhead=8, dim_feedforward=self.hidden_dim,
                                                  batch_first=True)
        self.netV = TransformerEncoder(encoder_layer_V, num_layers=1)
        encoder_layer_L = TransformerEncoderLayer(d_model=self.text_dim, nhead=8, dim_feedforward=self.hidden_dim,
                                                  batch_first=True)
        self.netL = TransformerEncoder(encoder_layer_L, num_layers=1)

        self.netC = FcClassifier(self.hidden_dim * 3, self.cls_layers, output_dim=6,
                                 dropout=self.dropout, use_bn=False)

        self.retrievaler = Retrieval_augment(k=6, dataset=self.dataset, datascale='MER_UNI')

        if self.isTrain:
            self.load_pretrained_encoder(opt)
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim
            self.ce_weight = opt.ce_weight
            self.mse_weight = opt.mse_weight
            self.invariant_weight = opt.invariant_weight
            self.cycle_weight = opt.cycle_weight

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        image_save_dir = os.path.join(opt.image_dir, opt.name)
        image_save_dir = os.path.join(image_save_dir, str(opt.cvNo))
        self.predict_image_save_dir = os.path.join(image_save_dir, 'predict')
        self.invariant_image_save_dir = os.path.join(image_save_dir, 'invariant')
        self.loss_image_save_dir = os.path.join(image_save_dir, 'loss')
        if not os.path.exists(self.predict_image_save_dir):
            os.makedirs(self.predict_image_save_dir)
        if not os.path.exists(self.invariant_image_save_dir):
            os.makedirs(self.invariant_image_save_dir)
        if not os.path.exists(self.loss_image_save_dir):
            os.makedirs(self.loss_image_save_dir)

    def load_pretrained_encoder(self, opt):
        print('Init parameter from {}'.format(opt.pretrained_path))
        pretrained_path = ''
        self.pretrained_encoder = CrossEncoder()
        state_dict = torch.load(pretrained_path)
        self.pretrained_encoder.load_state_dict(state_dict, strict=False)
        self.pretrained_encoder.cuda(self.device)



    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        # print(input['label'])
        self.acoustic = acoustic = input['A_feat'].float().to(self.device)
        self.lexical = lexical = input['L_feat'].float().to(self.device)
        self.visual = visual = input['V_feat'].float().to(self.device)
        self.label = input['label'].to(self.device)

        if self.isTrain:
            self.label = input['label'].to(self.device)
            self.missing_index = input['missing_index'].long().to(self.device)  # [a,v,l]
            # A modality
            self.A_miss_index = self.missing_index[:, 0].unsqueeze(1)
            self.A_miss = acoustic * self.A_miss_index
            # V modality
            self.V_miss_index = self.missing_index[:, 1].unsqueeze(1)
            self.V_miss = visual * self.V_miss_index
            # L modality
            self.L_miss_index = self.missing_index[:, 2].unsqueeze(1)
            self.L_miss = lexical * self.L_miss_index

        else:
            self.A_miss = acoustic
            self.V_miss = visual
            self.L_miss = lexical
            self.missing_index = input['missing_index'].long().to(self.device)  # [a,v,l]
            self.A_miss_index = self.missing_index[:, 0].unsqueeze(1)
            self.V_miss_index = self.missing_index[:, 1].unsqueeze(1)
            self.L_miss_index = self.missing_index[:, 2].unsqueeze(1)



    def forward(self):
        if self.isTrain:
            self.pretrained_encoder.train()
        else:
            self.pretrained_encoder.eval()
        b = self.A_miss.size(0)
        # input size = [b, e]
        self.cls_token_A1 = self.cls_token_A.expand(b, -1, -1)
        self.cls_token_V1 = self.cls_token_V.expand(b, -1, -1)
        self.cls_token_L1 = self.cls_token_L.expand(b, -1, -1)
        self.feat_A_miss = torch.cat((self.cls_token_A1, self.A_miss.unsqueeze(1)), dim=1)
        self.feat_V_miss = torch.cat((self.cls_token_V1, self.V_miss.unsqueeze(1)), dim=1)
        self.feat_L_miss = torch.cat((self.cls_token_L1, self.L_miss.unsqueeze(1)), dim=1)

        self.feat_A_miss = self.feat_A_miss + self.positional_encoding_A[:, :self.feat_A_miss.size(1), :]
        self.feat_V_miss = self.feat_V_miss + self.positional_encoding_V[:, :self.feat_V_miss.size(1), :]
        self.feat_L_miss = self.feat_L_miss + self.positional_encoding_L[:, :self.feat_L_miss.size(1), :]

        self.feat_A_miss = self.netA(self.feat_A_miss)
        self.feat_V_miss = self.netV(self.feat_V_miss)
        self.feat_L_miss = self.netL(self.feat_L_miss)

        self.feat_A_miss = self.netlinearA(self.A_miss)
        self.feat_V_miss = self.netlinearV(self.V_miss)
        self.feat_L_miss = self.netlinearT(self.L_miss)

        cls_output_A = self.feat_A_miss[:, 0, :]
        cls_output_V = self.feat_V_miss[:, 0, :]
        cls_output_L = self.feat_L_miss[:, 0, :]

        cls_output_A, cls_output_V, cls_output_L, self.count, self.total = \
            self.retrieval_augment(cls_output_A, cls_output_V, cls_output_L)


        self.feat_fusion_miss = torch.cat([cls_output_A, cls_output_V, cls_output_L],
                                          dim=1)

        self.logits, _ = self.netC(self.feat_fusion_miss)
        self.pred = F.softmax(self.logits, dim=-1)

    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_CE = self.ce_weight * self.criterion_ce(self.logits, self.label.long())


        loss = self.loss_CE
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # backward
        self.optimizer.zero_grad()
        self.backward()

        self.optimizer.step()

    def retrieval_augment(self, a, v, t):
        batch_size = self.A_miss.size(0)
        # final = torch.empty((self.A_miss.size(0), 512), dtype=torch.float32, device=self.device)
        # 预处理需要更新的索引
        A_miss_0 = (self.A_miss_index == 0)
        V_miss_0 = (self.V_miss_index == 0)
        L_miss_0 = (self.L_miss_index == 0)
        count = 0
        for i in range(batch_size):
            # 确定要搜索的类型和查询向量
            search_type = None
            query_vector1 = None
            query_vector2 = None
            if A_miss_0[i] and V_miss_0[i]:
                search_type = 'L'
                query_vector1 = t[i]
            elif A_miss_0[i] and L_miss_0[i]:
                search_type = 'V'
                query_vector1 = v[i]
            elif V_miss_0[i] and L_miss_0[i]:
                search_type = 'A'
                query_vector1 = a[i]
            elif not A_miss_0[i] and not V_miss_0[i] and L_miss_0[i]:
                search_type = 'AV'
                query_vector1 = a[i]
                query_vector2 = v[i]
            elif not A_miss_0[i] and V_miss_0[i] and not L_miss_0[i]:
                search_type = 'AL'
                query_vector1 = a[i]
                query_vector2 = t[i]
            elif A_miss_0[i] and not V_miss_0[i] and not L_miss_0[i]:
                search_type = 'VL'
                query_vector1 = v[i]
                query_vector2 = t[i]

            if search_type is not None:
                result, origin, sample, distance, search_type = self.retrievaler.get_most_similar_vectors(query_vector1,
                                                                                                          query_vector2,
                                                                                                          search_type)


            hidden_A = torch.tensor(np.array(result[0]), dtype=torch.float32, device=self.device)
            hidden_V = torch.tensor(np.array(result[1]), dtype=torch.float32, device=self.device)
            hidden_L = torch.tensor(np.array(result[2]), dtype=torch.float32, device=self.device)
            if search_type == 'L':
                a[i] = l2_normalize(sum(hidden_A))
                v[i] = l2_normalize(sum(hidden_V))
            elif search_type == 'V':
                a[i] = l2_normalize(sum(hidden_A))
                t[i] = l2_normalize(sum(hidden_L))

            elif search_type == 'A':
                v[i] = l2_normalize(sum(hidden_V))
                t[i] = l2_normalize(sum(hidden_L))

            elif search_type == 'AV':

                t[i] = l2_normalize(sum(hidden_L))

            elif search_type == 'AL':
                v[i] = l2_normalize(sum(hidden_V))

            elif search_type == 'VL':
                a[i] = l2_normalize(sum(hidden_A))



        return a, v, t, count, self.label.size(0)


def l2_normalize(tensor):
    norm = torch.norm(tensor, p=2)
    return tensor / norm




