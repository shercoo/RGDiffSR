import argparse, os, sys, datetime, glob, importlib, csv
import json
import string
from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np
import time

import ptflops
import torch
import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from text_super_resolution.utils.util import str_filt
from text_super_resolution.utils.labelmaps import get_vocabulary
from text_super_resolution.model import recognizer, crnn, moran
from text_super_resolution.utils.metrics import get_string_aster, get_string_crnn
from text_super_resolution.utils import ssim_psnr
from utils import utils_moran
from einops import rearrange
import matplotlib.pyplot as plt


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False, train_align_collate_fn=None, val_align_collate_fn=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn

        if train_align_collate_fn is not None:
            self.train_align_collate_fn = instantiate_from_config(train_align_collate_fn)
        else:
            self.train_align_collate_fn = None

        if val_align_collate_fn is not None:
            self.val_align_collate_fn = instantiate_from_config(val_align_collate_fn)
        else:
            self.val_align_collate_fn = None

        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn, collate_fn=self.train_align_collate_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle, collate_fn=self.val_align_collate_fn)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


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


class RecognizeCallback(Callback):
    def __init__(self, gpus, rec_interval):
        save_dir = os.getcwd() + '/logs/' + datetime.datetime.now().strftime('%Y-%m-%dT%H-%M_test')
        self.save_dir = {}
        for t in ['ASTER', 'MORAN', 'CRNN']:
            self.save_dir[t] = os.path.join(save_dir, t)
            os.makedirs(os.path.join(self.save_dir[t], 'correct'), exist_ok=True)
            os.makedirs(os.path.join(self.save_dir[t], 'wrong'), exist_ok=True)

        self.save_dir['metrics'] = os.path.join(save_dir, 'metrics')
        os.makedirs(self.save_dir['metrics'], exist_ok=True)
        print(self.save_dir)

        self.epoch_cnt = 0
        self.rec_interval = rec_interval
        self.voc_type = 'all'
        gpus = list(map(int, filter(lambda x: x != '', gpus.split(','))))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alphabet_moran = ':'.join(string.digits + string.ascii_lowercase + '$')
        self.converter_moran = utils_moran.strLabelConverterForAttention(alphabet_moran, ':')

        self.aster_info = AsterInfo(self.voc_type)
        aster_real = self.Aster_init(gpus)
        self.aster = [{
            'model': aster_real,
            'data_in_fn': self.parse_aster_data,
            'string_process': get_string_aster
        }]

        moran = self.MORAN_init(gpus)
        if isinstance(moran, torch.nn.DataParallel):
            moran.device_ids = [0]
        self.moran = [{
            'model': moran,
            'data_in_fn': self.parse_moran_data,
            'string_process': get_string_crnn
        }]

        crnn = self.CRNN_init()
        crnn.eval()
        self.crnn = [{
            'model': crnn,
            'data_in_fn': self.parse_crnn_data,
            'string_process': get_string_crnn
        }]

        self.cal_psnr = ssim_psnr.calculate_psnr
        self.cal_ssim = ssim_psnr.SSIM()
        self.recorders = {'CRNN': self.Recorder(),
                          'MORAN': self.Recorder(),
                          'ASTER': self.Recorder()}

    class Recorder(object):
        def __init__(self, ):
            self.n_correct = 0
            self.n_correct_lr = 0
            self.n_correct_hr = 0
            self.sum_images = 0
            self.metric_dict = {}
            self.image_counter = 0
            self.false_cnt = 0
            self.metric_init()

        def metric_init(self):
            self.n_correct = 0
            self.n_correct_lr = 0
            self.n_correct_hr = 0
            self.sum_images = 0
            self.false_cnt = 0
            self.metric_dict = {
                'psnr_lr': [],
                'ssim_lr': [],
                'cnt_psnr_lr': [],
                'cnt_ssim_lr': [],
                'psnr': [],
                'ssim': [],
                'cnt_psnr': [],
                'cnt_ssim': [],
                'accuracy': 0.0,
                'psnr_avg': 0.0,
                'ssim_avg': 0.0,
                'edis_LR': [],
                'edis_SR': [],
                'edis_HR': [],
                'LPIPS_VGG_LR': [],
                'LPIPS_VGG_SR': []
            }
            self.image_counter = 0

    def Aster_init(self, gpus):
        aster = recognizer.RecognizerBuilder(arch='ResNet_ASTER', rec_num_classes=self.aster_info.rec_num_classes,
                                             sDim=512, attDim=512, max_len_labels=self.aster_info.max_len,
                                             eos=self.aster_info.char2id[self.aster_info.EOS], STN_ON=True)
        aster_ckpt_path = 'aster.pth.tar'
        aster.load_state_dict(torch.load(aster_ckpt_path)['state_dict'])
        print('load pred_trained aster model from %s' % aster_ckpt_path)
        aster = aster.to(self.device)
        aster = torch.nn.DataParallel(aster, device_ids=gpus)
        aster.eval()
        return aster

    def MORAN_init(self, gpus):

        alphabet = ':'.join(string.digits + string.ascii_lowercase + '$')
        MORAN = moran.MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
                            inputDataType='torch.cuda.FloatTensor', CUDA=True)
        model_path = 'moran.pth'
        print('loading pre-trained moran model from %s' % model_path)
        state_dict = torch.load(model_path)
        MORAN_state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # remove `module.`
            MORAN_state_dict_rename[name] = v
        MORAN.load_state_dict(MORAN_state_dict_rename)
        MORAN = MORAN.to(self.device)
        MORAN = torch.nn.DataParallel(MORAN, device_ids=gpus)
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
        return MORAN

    def CRNN_init(self, recognizer_path='crnn.pth', imgH=32):
        model = crnn.CRNN(imgH, 1, 37, 256)
        model = model.to(self.device)

        macs, params = ptflops.get_model_complexity_info(model, (1, 32, 100), as_strings=True,
                                                         print_per_layer_stat=False, verbose=True)
        print("---------------- TP Module -----------------")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print("--------------------------------------------")

        print("recognizer_path:", recognizer_path)

        if recognizer_path is not None:
            model_path = recognizer_path
            print('loading pretrained crnn model from %s' % model_path)
            stat_dict = torch.load(model_path)
            if type(stat_dict) == OrderedDict:
                print("The dict:")
                model.load_state_dict(stat_dict)
            else:
                print("The model:")
                model = stat_dict
        model.eval()
        return model

    def parse_aster_data(self, imgs_input):
        input_dict = {}
        images_input = imgs_input.to(self.device)
        input_dict['images'] = images_input * 2 - 1
        batch_size = images_input.shape[0]
        input_dict['rec_targets'] = torch.IntTensor(batch_size, self.aster_info.max_len).fill_(1)
        input_dict['rec_lengths'] = [self.aster_info.max_len] * batch_size
        return input_dict

    def parse_moran_data(self, imgs_input):
        batch_size = imgs_input.shape[0]

        # in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100
        in_width = 100

        imgs_input = torch.nn.functional.interpolate(imgs_input, (32, in_width), mode='bicubic')
        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        text = torch.LongTensor(batch_size * 5)
        length = torch.IntTensor(batch_size)
        max_iter = 20
        t, l = self.converter_moran.encode(['0' * max_iter] * batch_size)
        utils_moran.loadData(text, t)
        utils_moran.loadData(length, l)
        return tensor, length, text, text

    def parse_crnn_data(self, imgs_input_, ratio_keep=False):

        # in_width = self.config.TRAIN.width if self.config.TRAIN.width != 128 else 100
        in_width = 100

        if ratio_keep:
            real_height, real_width = imgs_input_.shape[2:]
            ratio = real_width / float(real_height)

            if ratio > 3:
                in_width = int(ratio * 32)
        imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic')

        # print("imgs_input:", imgs_input.shape)

        R = imgs_input[:, 0:1, :, :]
        G = imgs_input[:, 1:2, :, :]
        B = imgs_input[:, 2:3, :, :]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        return tensor

    def recognize(self, pl_module, batch, batch_idx, split="train"):
        print('******************************recognize*********************************')
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.recognize_sample(batch, N=114514, split=split, inpaint=False)

        images_sr = images['samples']
        # latent=images['latent']
        #
        # cvimg = cv2.cvtColor(latent[0, :3, :, :].cpu().numpy().transpose(1, 2, 0) * 255,
        #                      cv2.COLOR_RGB2BGR)
        # cv2.imwrite('/home/zhouyuxuan/latent.jpg'
        #             , cvimg)
        # exit(0)

        # for k in images:
        #     N = min(images[k].shape[0], self.max_images)
        #     images[k] = images[k][:N]
        #     if isinstance(images[k], torch.Tensor):
        #         images[k] = images[k].detach().cpu()
        #         if self.clamp:
        #             images[k] = torch.clamp(images[k], -1., 1.)

        if is_train:
            pl_module.train()

        return images_sr

    def eval(self, batch, images_sr, aster, aster_info, test_model_type):

        #############################################
        # Print the computational cost and param size
        # self.cal_all_models(model_list, aster[1])
        #############################################

        images_hr = batch['image']
        images_lr = batch['LR_image']
        label_strs = batch['label']
        indexes = batch['id']

        images_lr = rearrange(images_lr, 'b h w c -> b c h w')
        images_hr = rearrange(images_hr, 'b h w c -> b c h w')

        images_lr = images_lr.to(self.device)
        images_hr = images_hr.to(self.device)

        val_batch_size = images_lr.shape[0]

        # print("images_lr:", images_lr.device, images_hr.device)

        aster_dict_lr = aster[0]["data_in_fn"](images_lr[:, :3, :, :])
        aster_dict_hr = aster[0]["data_in_fn"](images_hr[:, :3, :, :])
        aster_dict_sr = aster[0]["data_in_fn"](images_sr[:, :3, :, :])

        if test_model_type == "MORAN":
            # aster_output_sr = aster[0]["model"](*aster_dict_sr)
            # LR
            aster_output_lr = aster[0]["model"](
                aster_dict_lr[0],
                aster_dict_lr[1],
                aster_dict_lr[2],
                aster_dict_lr[3],
                test=True,
                debug=True
            )
            # HR
            aster_output_hr = aster[0]["model"](
                aster_dict_hr[0],
                aster_dict_hr[1],
                aster_dict_hr[2],
                aster_dict_hr[3],
                test=True,
                debug=True
            )
            aster_output_sr = aster[0]["model"](
                aster_dict_sr[0],
                aster_dict_sr[1],
                aster_dict_sr[2],
                aster_dict_sr[3],
                test=True,
                debug=True
            )
        else:
            aster_output_lr = aster[0]["model"](aster_dict_lr)
            aster_output_hr = aster[0]["model"](aster_dict_hr)
            aster_output_sr = aster[0]["model"](aster_dict_sr)

        # aster_dict_sr = aster[0]["data_in_fn"](images_sr[:, :3, :, :])
        #
        # aster_output_sr = aster[0]["model"](aster_dict_sr)
        # outputs_sr = aster_output_sr.permute(1, 0, 2).contiguous()

        if test_model_type == "CRNN":
            predict_result_sr = aster[0]["string_process"](aster_output_sr, False)
            predict_result_hr = aster[0]["string_process"](aster_output_hr, False)
            predict_result_lr = aster[0]["string_process"](aster_output_lr, False)
        elif test_model_type == "ASTER":
            predict_result_sr, _ = aster[0]["string_process"](
                aster_output_sr['output']['pred_rec'],
                aster_dict_sr['rec_targets'],
                dataset=aster_info
            )
            predict_result_lr, _ = aster[0]["string_process"](
                aster_output_lr['output']['pred_rec'],
                aster_dict_lr['rec_targets'],
                dataset=aster_info
            )
            predict_result_hr, _ = aster[0]["string_process"](
                aster_output_hr['output']['pred_rec'],
                aster_dict_hr['rec_targets'],
                dataset=aster_info
            )

        elif test_model_type == "MORAN":
            preds, preds_reverse = aster_output_sr[0]
            _, preds = preds.max(1)
            sim_preds = self.converter_moran.decode(preds.data, aster_dict_sr[1].data)
            predict_result_sr = [pred.split('$')[0] for pred in sim_preds]

            preds, preds_reverse = aster_output_hr[0]
            _, preds = preds.max(1)
            sim_preds = self.converter_moran.decode(preds.data, aster_dict_hr[1].data)
            predict_result_hr = [pred.split('$')[0] for pred in sim_preds]

            preds, preds_reverse = aster_output_lr[0]
            _, preds = preds.max(1)
            sim_preds = self.converter_moran.decode(preds.data, aster_dict_lr[1].data)
            predict_result_lr = [pred.split('$')[0] for pred in sim_preds]

        # predict_result_sr, _ = aster[0]["string_process"](
        #     aster_output_sr['output']['pred_rec'],
        #     aster_dict_sr['rec_targets'],
        #     dataset=aster_info
        # )

        img_lr = torch.nn.functional.interpolate(images_lr, images_sr.shape[-2:], mode="bicubic")

        self.recorders[test_model_type].metric_dict['psnr'].append(self.cal_psnr(images_sr[:, :3], images_hr[:, :3]))
        self.recorders[test_model_type].metric_dict['ssim'].append(self.cal_ssim(images_sr[:, :3], images_hr[:, :3]))

        self.recorders[test_model_type].metric_dict['psnr_lr'].append(self.cal_psnr(img_lr[:, :3], images_hr[:, :3]))
        self.recorders[test_model_type].metric_dict['ssim_lr'].append(self.cal_ssim(img_lr[:, :3], images_hr[:, :3]))

        filter_mode = 'lower'

        for batch_i in range(images_lr.shape[0]):
            label = label_strs[batch_i]
            # print(predict_result_sr[batch_i],predict_result_lr[batch_i],predict_result_hr[batch_i],label)
            self.recorders[test_model_type].image_counter += 1

            plt.figure()
            plt.title(label)
            plt.subplot(1, 3, 1)
            plt.imshow(images_lr[batch_i, :3, :, :].cpu().numpy().transpose(1, 2, 0))
            plt.title(predict_result_lr[batch_i])
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(images_hr[batch_i, :3, :, :].cpu().numpy().transpose(1, 2, 0))
            plt.title(predict_result_hr[batch_i])
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(images_sr[batch_i, :3, :, :].cpu().numpy().transpose(1, 2, 0))
            plt.title(predict_result_sr[batch_i])
            plt.axis('off')

            if str_filt(predict_result_sr[batch_i], filter_mode) == str_filt(label, filter_mode):
                self.recorders[test_model_type].n_correct += 1
                plt.savefig(os.path.join(self.save_dir[test_model_type], f'correct/{indexes[batch_i]}_comp.jpg'))

                cvimg = cv2.cvtColor(images_sr[batch_i, :3, :, :].cpu().numpy().transpose(1, 2, 0) * 255,
                                     cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.save_dir[test_model_type], f'correct/{indexes[batch_i]}.jpg')
                            , cvimg)

            else:
                self.recorders[test_model_type].false_cnt += 1
                plt.savefig(os.path.join(self.save_dir[test_model_type], f'wrong/{indexes[batch_i]}_comp.jpg'))

                cvimg = cv2.cvtColor(images_sr[batch_i, :3, :, :].cpu().numpy().transpose(1, 2, 0) * 255,
                                     cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.save_dir[test_model_type], f'wrong/{indexes[batch_i]}.jpg')
                            , cvimg)

                # print(os.path.join(self.save_dir, f'{self.false_cnt}.jpg'))

            if str_filt(predict_result_lr[batch_i], filter_mode) == str_filt(label, filter_mode):
                self.recorders[test_model_type].n_correct_lr += 1

            if str_filt(predict_result_hr[batch_i], filter_mode) == str_filt(label, filter_mode):
                self.recorders[test_model_type].n_correct_hr += 1

        self.recorders[test_model_type].sum_images += val_batch_size
        torch.cuda.empty_cache()
        # print(f'sr correct:{self.n_correct}/{self.sum_images}')
        # print(f'lr correct:{self.n_correct_lr}/{self.sum_images}')
        # print(f'hr correct:{self.n_correct_hr}/{self.sum_images}')

    def show_results(self, pl_module, test_model_type):

        rec = self.recorders[test_model_type]
        psnr_avg = sum(rec.metric_dict['psnr']) / (len(rec.metric_dict['psnr']) + 1e-10)
        ssim_avg = sum(rec.metric_dict['ssim']) / (len(rec.metric_dict['ssim']) + 1e-10)

        psnr_avg_lr = sum(rec.metric_dict['psnr_lr']) / (len(rec.metric_dict['psnr_lr']) + 1e-10)
        ssim_avg_lr = sum(rec.metric_dict['ssim_lr']) / (len(rec.metric_dict['ssim_lr']) + 1e-10)

        print('[{}]\t'
              'PSNR {:.2f} | SSIM {:.4f}\t'
              .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(psnr_avg), float(ssim_avg)))

        print('[{}]\t'
              'PSNR_LR {:.2f} | SSIM_LR {:.4f}\t'
              .format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      float(psnr_avg_lr), float(ssim_avg_lr)))

        # self.tripple_display(images_lr, images_sr, images_hr, pred_str_lr, pred_str_sr, label_strs, index)

        accuracy = round(rec.n_correct / rec.sum_images, 4)
        accuracy_lr = round(rec.n_correct_lr / rec.sum_images, 4)
        accuracy_hr = round(rec.n_correct_hr / rec.sum_images, 4)
        psnr_avg = round(psnr_avg.item(), 6)
        ssim_avg = round(ssim_avg.item(), 6)

        print('sr_accuracy: %.2f%%' % (accuracy * 100))

        # print('sr_NED: %.4f' % (edis_SR))
        print('lr_accuracy: %.2f%%' % (accuracy_lr * 100))
        # print('lr_NED: %.4f' % (edis_LR))
        print('hr_accuracy: %.2f%%' % (accuracy_hr * 100))
        # print('hr_NED: %.4f' % (edis_HR))
        log_dict = {f'{test_model_type}_accuracy': accuracy, 'psnr_avg': psnr_avg, 'ssim_avg': ssim_avg}

        with open(os.path.join(self.save_dir['metrics'], f'{torch.cuda.current_device()}.json'), 'w') as f:
            json.dump(log_dict, f)
        pl_module.log_dict(log_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        print("sum_images:", rec.sum_images)
        self.recorders[test_model_type].metric_init()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if (self.epoch_cnt + 1) % self.rec_interval == 0:
            images_sr = self.recognize(pl_module, batch, batch_idx, split="val")
            self.eval(batch, images_sr, self.aster, self.aster_info, 'ASTER')
            self.eval(batch, images_sr, self.moran, self.aster_info, 'MORAN')
            self.eval(batch, images_sr, self.crnn, self.aster_info, 'CRNN')

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if (self.epoch_cnt + 1) % self.rec_interval == 0:
            self.show_results(pl_module, 'ASTER')
            self.show_results(pl_module,'MORAN')
            self.show_results(pl_module,'CRNN')

    def on_train_epoch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", unused: Optional = None
    ):
        self.epoch_cnt += 1


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["accelerator"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
                "every_n_epochs": 50
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "increase_log_steps": False,
                    "batch_frequency": 2000,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
            "recognize_callback": {
                "target": "test.RecognizeCallback",
                "params": {
                    "gpus": lightning_config.trainer.gpus,
                    "rec_interval": 1
                }
            },

        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                     'params': {
                         "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                         "filename": "{epoch:06}-{step:09}",
                         "verbose": True,
                         'save_top_k': -1,
                         'every_n_train_steps': 10000,
                         'save_weights_only': True
                     }
                     }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        print('**************trainer***************')
        print(trainer_opt)
        print(trainer_kwargs)
        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###
        trainer.check_val_every_n_epoch = 1

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        print("#### Data #####")
        for k in data.datasets:
            print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)
        #

        total = sum([param.nelement() for param in model.parameters()])
        print('Number of parameter: % .4fM' % (total / 1e6))


        trainer.validate(model, data)
        exit(0)

    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
