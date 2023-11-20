# coding:utf-8
import io
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import pdb
import os
import cv2
from tqdm import tqdm

from einops import rearrange, repeat

sys.path.append('./')
sys.path.append('/home/zhouyuxuan/latent-diffusion/')
from ldm.data.transforms import CVColorJitter, CVDeterioration, CVGeometry
import re
from random import sample
from text_super_resolution.utils import ssim_psnr


def des_orderlabel(imput_lable):
    '''
    generate the label for WCL
    '''
    if True:
        len_str = len(imput_lable)
        change_num = 1
        order = list(range(len_str))
        change_id = sample(order, change_num)[0]
        label_sub = imput_lable[change_id]
        if change_id == (len_str - 1):
            imput_lable = imput_lable[:change_id]
        else:
            imput_lable = imput_lable[:change_id] + imput_lable[change_id + 1:]
    return imput_lable, label_sub, change_id


class lmdbDataset(Dataset):
    def __init__(self, roots=None, ratio=None, img_height=32, img_width=128,
                 transform=None, global_state='Test'):
        self.envs = []
        self.nSamples = 0
        self.lengths = []
        self.ratio = []
        self.global_state = global_state
        for i in range(0, len(roots)):
            env = lmdb.open(
                roots[i],
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)
            if not env:
                print('cannot creat lmdb from %s' % (roots[i]))
                sys.exit(0)
            with env.begin(write=False) as txn:
                nSamples = int(txn.get('num-samples'.encode()))
                self.nSamples += nSamples
            self.lengths.append(nSamples)
            self.envs.append(env)

        if ratio != None:
            assert len(roots) == len(ratio), 'length of ratio must equal to length of roots!'
            for i in range(0, len(roots)):
                self.ratio.append(ratio[i] / float(sum(ratio)))
        else:
            for i in range(0, len(roots)):
                self.ratio.append(self.lengths[i] / float(self.nSamples))
        self.transform = transform
        self.maxlen = max(self.lengths)
        self.img_height = img_height
        self.img_width = img_width
        self.target_ratio = img_width / float(img_width)
        self.min_size = (img_width * 0.5, img_width * 0.75, img_width)

        self.augment_tfs = transforms.Compose([
            CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
            CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
            CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
        ])

    def __fromwhich__(self):
        rd = random.random()
        total = 0
        for i in range(0, len(self.ratio)):
            total += self.ratio[i]
            if rd <= total:
                return i

    def keepratio_resize(self, img, is_train):
        if is_train == 'Train':
            img = self.augment_tfs(img)
        img = cv2.resize(np.array(img), (self.img_width, self.img_height))
        img = transforms.ToPILImage()(img)
        return img

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        fromwhich = self.__fromwhich__()
        if self.global_state == 'Train':
            index = random.randint(0, self.maxlen - 1)
        index = index % self.lengths[fromwhich]
        assert index <= len(self), 'index range error'
        index += 1
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf).convert('RGB')
            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()))
            # if python3
            # label = str(txn.get(label_key.encode()), 'utf-8')
            label = re.sub('[^0-9a-zA-Z]+', '', label)

            if (len(label) > 25 or len(label) <= 0) and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img = self.keepratio_resize(img, self.global_state)
            except:
                print('Size error for %d' % index)
                return self[index + 1]
            if self.transform:
                img = self.transform(img)
            # generate masked_id masked_character remain_string
            # label_res, label_sub, label_id = des_orderlabel(label)
            # sample = {'image': img, 'label': label, 'label_res': label_res, 'label_sub': label_sub,
            #           'label_id': label_id}
            sample = {'image': img, 'label': label}
            return sample


class occluded_lmdbDataset(lmdbDataset):
    def __getitem__(self, index):
        fromwhich = self.__fromwhich__()
        if self.global_state == 'Train':
            index = random.randint(0, self.maxlen - 1)
        index = index % self.lengths[fromwhich]
        assert index <= len(self), 'index range error'
        index += 1
        with self.envs[fromwhich].begin(write=False) as txn:
            img_key = 'image-%09d' % index
            try:
                imgbuf = txn.get(img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = Image.open(buf).convert('RGB')
            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            occ_img_key = 'occluded-image-%09d' % index
            try:
                imgbuf = txn.get(occ_img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                occ_img = Image.open(buf).convert('RGB')
            except:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()))
            # if python3
            # label = str(txn.get(label_key.encode()), 'utf-8')
            label = re.sub('[^0-9a-zA-Z]+', '', label)

            if (len(label) > 25 or len(label) <= 0) and self.global_state == 'Train':
                print('sample too long')
                return self[index + 1]
            try:
                img = self.keepratio_resize(img, self.global_state)
                occ_img = self.keepratio_resize(occ_img, self.global_state)
            except:
                print('Size error for %d' % index)
                return self[index + 1]

            if self.transform:
                img = self.transform(img)
            # generate masked_id masked_character remain_string
            # label_res, label_sub, label_id = des_orderlabel(label)
            # sample = {'image': img, 'label': label, 'label_res': label_res, 'label_sub': label_sub,
            #           'label_id': label_id}
            # sample = {'OC_image': occ_img, 'image': img, 'label': label}
            return occ_img, img, label


class alignCollate_occluded(object):
    def __init__(self, imgH=32,
                 imgW=128,
                 down_sample_scale=2,
                 keep_ratio=False,
                 min_ratio=1,
                 mask=False,
                 alphabet=53,
                 train=True,
                 y_domain=False
                 ):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.down_sample_scale = down_sample_scale
        self.toTensor = transforms.ToTensor()
        # self.mask = mask
        # self.alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        # self.alphabet = open("al_chinese.txt", "r").readlines()[0].replace("\n", "")
        self.train = train

    def transform(self, img, size, interpolation=Image.BICUBIC):
        img = img.resize(size, interpolation)
        img = self.toTensor(img)
        return img

    def __call__(self, batch):
        occ_imgs, imgs, label_strs = zip(*batch)
        imgH = self.imgH
        imgW = self.imgW
        # print(occ_imgs)

        occ_imgs = [self.transform(image, (imgW // self.down_sample_scale, imgH // self.down_sample_scale))
                    for image in occ_imgs]
        occ_imgs = torch.cat([t.unsqueeze(0) for t in occ_imgs], 0)

        imgs = [self.transform(image, (imgW, imgH)) for image in imgs]
        imgs = torch.cat([t.unsqueeze(0) for t in imgs], 0)

        imgs = rearrange(imgs, 'b c h w -> b h w c')
        occ_imgs = rearrange(occ_imgs, 'b c h w -> b h w c')

        example = {'image': imgs, 'OC_image': occ_imgs}
        return example


def write_cache(env, cache):
    txn = env.begin(write=True)
    for k, v in cache.items():
        txn.put(k, v)
    txn.commit()


def select_lmdb_data(raw_dataset, output_dir, output_size, min_text_len=5, max_text_len=50):
    os.makedirs(output_dir, exist_ok=True)
    env = lmdb.open(output_dir, map_size=1099511627776)
    cache = {}
    valid_num = 0
    raw_idxs = list(range(len(raw_dataset)))
    pbar = tqdm(total=output_size)
    while valid_num < output_size:
        random.shuffle(raw_idxs)
        for data_idx in raw_idxs:
            occ_img, img, label = raw_dataset[data_idx]
            if min_text_len <= len(label) <= max_text_len:
                valid_num += 1
                buff = io.BytesIO()
                occ_buff = io.BytesIO()
                img.save(buff, format='PNG')
                occ_img.save(occ_buff, format='PNG')

                image_key = 'image-%09d'.encode() % valid_num
                occ_image_key = 'occluded-image-%09d'.encode() % valid_num
                label_key = 'label-%09d'.encode() % valid_num

                cache[image_key] = buff.getvalue()
                cache[occ_image_key] = occ_buff.getvalue()
                cache[label_key] = label.encode()

                if valid_num % 1000 == 0:
                    write_cache(env, cache)
                    cache = {}
                    # print('Written %d / %d' % (valid_num, output_size))
                    pbar.update(1000)

            if valid_num >= output_size:
                break
    n_samples = valid_num
    cache['num-samples'.encode()] = str(n_samples).encode()
    write_cache(env, cache)
    print('Created dataset with %d samples' % n_samples)


def calc_similarity(img1, img2,wtf=False):
    # print(img1.size,img2.size)
    # print(np.array(img1).shape)
    # exit(0)
    if wtf:
        print(np.abs(np.array(img1.convert('L')) - np.array(img2.convert('L'))))
    return np.abs(np.array(img1.convert('L')) - np.array(img2.convert('L'))).sum()

psnr = ssim_psnr.calculate_psnr
ssim = ssim_psnr.SSIM()

def calc_ssim(img1,img2):
    img1=np.expand_dims(np.array(img1).transpose((2,0,1)),axis=0)
    img1=torch.Tensor(img1)
    img2=np.expand_dims(np.array(img2).transpose((2,0,1)),axis=0)
    img2=torch.Tensor(img2)
    # print(img1.size())
    return ssim(img1,img2)


if __name__ == '__main__':
    # occ=occluded_lmdbDataset(['/home/zhouyuxuan/OST/weak_pair'])
    # sample = occ.__getitem__(2)
    # img = sample['image']
    # occ_img=sample['OC_image']
    # img.save('/home/zhouyuxuan/ori.jpg')
    # occ_img.save('/home/zhouyuxuan/occ.jpg')
    # exit(0)

    random.seed(0)
    all = []
    labelmap = {}
    STR = lmdbDataset(roots=['/home/zhouyuxuan/OST/Sumof6benchmarks'])
    print(len(STR))
    for id in tqdm(range(len(STR))):
        sample = STR.__getitem__(id)
        img = sample['image']
        label = sample['label']
        all.append((img, label))
        if labelmap.get(label) is None:
            labelmap[label] = [id]
        else:
            labelmap[label].append(id)

    raw_dataset = []
    cnt = 0
    for path in ['/home/zhouyuxuan/OST/weak', '/home/zhouyuxuan/OST/heavy']:
        OST = lmdbDataset(roots=[path])

        for id in tqdm(range(len(OST))):
            sample = OST.__getitem__(id)
            img = sample['image']
            label = sample['label']
            if labelmap.get(label) is None:
                assert False, 'wtf'
            else:
                # simid = np.array([calc_similarity(img, all[id][0]) for id in labelmap[label]]).argmin()
                simid = np.array([calc_ssim(img, all[id][0]) for id in labelmap[label]]).argmax()
                simid = labelmap[label][simid]
                raw_dataset.append((img, all[simid][0], label))
                if len(labelmap[label]) >= 2:
                    if random.randint(1, 20) == 1:
                        cnt += 1
                    if cnt == 1:
                        print(label)
                        for i in range(len(labelmap[label])):
                            if label == 'bSBI' :
                                print(i,calc_ssim(img, all[labelmap[label][i]][0]))
                            all[labelmap[label][i]][0].save(
                                f'/home/zhouyuxuan/tmp/fig{i}_{all[labelmap[label][i]][1]}.jpg')
                    if cnt <= 5:
                        img.save(f'/home/zhouyuxuan/tmp/occ{cnt}.jpg')
                        all[simid][0].save(f'/home/zhouyuxuan/tmp/ori{cnt}.jpg')

    print(cnt)

    random.shuffle(raw_dataset)
    tot = len(raw_dataset)
    select_lmdb_data(raw_dataset[:tot - tot // 10], '/home/zhouyuxuan/OST/occluded/train/', tot - tot // 10,
                     min_text_len=1)
    select_lmdb_data(raw_dataset[tot - tot // 10:], '/home/zhouyuxuan/OST/occluded/test/', tot // 10, min_text_len=1)
