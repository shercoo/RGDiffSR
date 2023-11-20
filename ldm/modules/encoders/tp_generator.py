import datetime
import math
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
import sys
from torch.nn import init
import numpy as np
from IPython import embed

from text_super_resolution.model.transformer_v2 import InfoTransformer
from text_super_resolution.model.transformer_v2 import PositionalEncoding

import ptflops
from text_super_resolution.model import crnn
from text_super_resolution.utils.labelmaps import get_vocabulary


class AsterInfo(object):
    def __init__(self, voc_type):
        super(AsterInfo, self).__init__()
        self.voc_type = voc_type
        assert voc_type in ['digit', 'lower', 'upper', 'all', 'chinese']
        self.EOS = 'EOS'
        self.max_len = 100
        self.PADDING = 'PADDING'
        self.UNKNOWN = 'UNKNOWN'
        self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
        self.char2id = dict(zip(self.voc, range(len(self.voc))))
        self.id2char = dict(zip(range(len(self.voc)), self.voc))
        self.rec_num_classes = len(self.voc)


class TP_generator(nn.Module):

    def __init__(
            self,
            imgH,
            recognizer_path=None,
    ):
        super(TP_generator, self).__init__()
        self.imgH = imgH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.crnn_model, _ = self.CRNN_init(recognizer_path, imgH)
        self.timestamp=datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

    def CRNN_init(self, recognizer_path=None, imgH=32):
        model = crnn.CRNN(imgH, 1, 37, 256)
        model = model.to(self.device)

        macs, params = ptflops.get_model_complexity_info(model, (1, 32, 100), as_strings=True,
                                                         print_per_layer_stat=False, verbose=True)
        print("---------------- TP Module -----------------")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print("--------------------------------------------")

        print("recognizer_path:", recognizer_path)

        aster_info = AsterInfo('all')
        if recognizer_path is not None:
            model_path = recognizer_path
            print('loading pretrained crnn model from %s' % model_path)
            stat_dict = torch.load(model_path)
            # print("stat_dict:", stat_dict.keys())
            # if recognizer_path is None:
            #     model.load_state_dict(stat_dict)
            # else:
                # print("stat_dict:", stat_dict)
                # print("stat_dict:", type(stat_dict) == OrderedDict)
            if type(stat_dict) == OrderedDict:
                print("The dict:")
                model.load_state_dict(stat_dict)
            else:
                print("The model:")
                model = stat_dict
        # model #.eval()
        # model.eval()
        return model, aster_info

    def save_state_dict(self, path, epoch):
        torch.save(self.crnn_model.state_dict(), path+self.timestamp+f'-e{epoch}.pth')

    def parse_crnn_data(self, imgs_input_, ratio_keep=False):

        # in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100
        in_width = 100

        if ratio_keep:
            real_height, real_width = imgs_input_.shape[2:]
            ratio = real_width / float(real_height)

            if ratio > 3:
                in_width = int(ratio * 32)
        imgs_input = torch.nn.functional.interpolate(imgs_input_, (self.imgH, in_width), mode='bicubic')

        # print("imgs_input:", imgs_input.shape)

        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def forward(self, image):

        # H, W = self.output_size
        # x = tp_input #b,h,1,l
        image = self.parse_crnn_data(image)
        # print(image.shape)
        label_vecs_logits = self.crnn_model(image)
        label_vecs = torch.nn.functional.softmax(label_vecs_logits, -1)  # l,b,h

        # print(label_vecs.shape,len(label_strs),label_strs)
        # print(get_string_crnn(label_vecs,use_chinese=False))
        # print(get_string_crnn(label_vecs_hr,use_chinese=False))
        # exit(0)

        label_vecs_final = label_vecs.permute(1, 0, 2)  # b,l,h
        return label_vecs_final


class TPInterpreter(nn.Module):
    def __init__(
            self,
            t_emb,
            out_text_channels,
            output_size=(16, 64),
            feature_in=64,
            # d_model=512,
            t_encoder_num=1,
            t_decoder_num=2,
    ):
        super(TPInterpreter, self).__init__()

        d_model = out_text_channels  # * output_size[0]

        self.fc_in = nn.Linear(t_emb, d_model)
        self.fc_in2 = nn.Linear(4096, d_model)
        self.fc_feature_in = nn.Linear(feature_in, d_model)

        self.activation = nn.PReLU()

        self.transformer = InfoTransformer(d_model=d_model,
                                           dropout=0.1,
                                           nhead=4,
                                           dim_feedforward=d_model,
                                           num_encoder_layers=t_encoder_num,
                                           num_decoder_layers=t_decoder_num,
                                           normalize_before=False,
                                           return_intermediate_dec=True, feat_height=output_size[0],
                                           feat_width=output_size[1])

        self.pe = PositionalEncoding(d_model=d_model, dropout=0.1, max_len=5000)

        self.output_size = output_size
        self.seq_len = output_size[1] * output_size[0]  # output_size[1] ** 2 #
        self.init_factor = nn.Embedding(self.seq_len, d_model)

        self.masking = torch.ones(output_size)

        # self.tp_uper = InfoGen(t_emb, out_text_channels)

    def forward(self, image_feature, tp_input):
        # H, W = self.output_size
        x = tp_input  # b,h,1,l
        # print(x.shape)

        N_i, C_i, H_i, W_i = image_feature.shape
        H, W = H_i, W_i

        x_tar = image_feature

        # [1024, N, 64]
        x_im = x_tar.view(N_i, C_i, H_i * W_i).permute(2, 0, 1)

        device = x.device
        # print('x:{} s:{}'.format( x.shape,s.shape))

        x = x.permute(0, 3, 1, 2).squeeze(-1)  # b,l,h
        x = self.activation(self.fc_in(x))
        N, L, C = x.shape

        x_pos = self.pe(torch.zeros((N, L, C)).to(device)).permute(1, 0, 2)
        x_mask = torch.zeros((N, L)).to(device).bool()
        x = x.permute(1, 0, 2)  # l,b,h (26,b,1024)

        # print('fuck',x.shape)

        text_prior, pr_weights = self.transformer(x, x_mask, self.init_factor.weight, x_pos,
                                                  # s, s_mask, s_pos,
                                                  tgt=x_im, spatial_size=(H, W))  # self.init_factor.weight
        text_prior = text_prior.mean(0)
        # print(text_prior.shape)
        # exit(0)
        text_prior = text_prior.permute(1, 2, 0).view(N, C, H, W)

        return text_prior

# class TP_generator(nn.Module):
#     def __init__(self,
#                  scale_factor=2,
#                  width=128,
#                  height=32,
#                  STN=False,
#                  srb_nums=5,
#                  mask=True,
#                  hidden_units=32,
#                  word_vec_d=300,
#                  text_emb=37,  # 37, #26+26+1 3965
#                  out_text_channels=64,  # 32 256
#                  feature_rotate=False,
#                  rotate_train=3.):
#         super(TP_generator, self).__init__()
#         in_planes = 3
#         if mask:
#             in_planes = 4
#         assert math.log(scale_factor, 2) % 1 == 0
#         upsample_block_num = int(math.log(scale_factor, 2))
#         self.block1 = nn.Sequential(
#             nn.Conv2d(in_planes, 2 * hidden_units, kernel_size=9, padding=4),
#             nn.PReLU()
#         )
#
#         self.infoGen = TPInterpreter(text_emb, out_text_channels, output_size=(
#         height // scale_factor, width // scale_factor))  # InfoGen(text_emb, out_text_channels)
#
#         self.feature_rotate = feature_rotate
#         self.rotate_train = rotate_train
#
#         if not SHUT_BN:
#             setattr(self, 'block%d' % (srb_nums + 2),
#                     nn.Sequential(
#                         nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
#                         nn.BatchNorm2d(2 * hidden_units)
#                     ))
#         else:
#             setattr(self, 'block%d' % (srb_nums + 2),
#                     nn.Sequential(
#                         nn.Conv2d(2 * hidden_units, 2 * hidden_units, kernel_size=3, padding=1),
#                         # nn.BatchNorm2d(2 * hidden_units)
#                     ))
#
#         block_ = [UpsampleBLock(2 * hidden_units, 2) for _ in range(upsample_block_num)]
#         block_.append(nn.Conv2d(2 * hidden_units, in_planes, kernel_size=9, padding=4))
#         setattr(self, 'block%d' % (srb_nums + 3), nn.Sequential(*block_))
#         self.tps_inputsize = [height // scale_factor, width // scale_factor]
#         tps_outputsize = [height // scale_factor, width // scale_factor]
#         num_control_points = 20
#         tps_margins = [0.05, 0.05]
#         self.stn = STN
#         if self.stn:
#             self.tps = TPSSpatialTransformer(
#                 output_image_size=tuple(tps_outputsize),
#                 num_control_points=num_control_points,
#                 margins=tuple(tps_margins))
#
#             self.stn_head = STNHead(
#                 in_planes=in_planes,
#                 num_ctrlpoints=num_control_points,
#                 activation='none',
#                 input_size=self.tps_inputsize)
#
#         self.block_range = [k for k in range(2, self.srb_nums + 2)]
#
#     # print("self.block_range:", self.block_range)
#
#     def forward(self, x, text_emb=None, text_emb_gt=None, feature_arcs=None, rand_offs=None, stroke_map=None):
#
#         if self.stn and self.training:
#             _, ctrl_points_x = self.stn_head(x)
#             x, _ = self.tps(x, ctrl_points_x)
#         block = {'1': self.block1(x)}
#
#         if text_emb is None:
#             text_emb = torch.zeros(1, 37, 1, 26).to(x.device)  # 37
#         if stroke_map is None:
#             stroke_map = torch.zeros(1, 16, 26, 256).to(x.device)
#         padding_feature = block['1']
#
#         tp_map_gt, pr_weights_gt = None, None
#         # print('text_emb:{} stroke_map:{}'.format( text_emb.shape,stroke_map.shape))
#         tp_map, pr_weights = self.infoGen(padding_feature, text_emb, stroke_map)
#         # N, C, H, W
