import sys

import numpy as np
from PIL import Image

sys.path.append('../')
from ldm.data.textzoom import lmdbDataset_real as textzoom
from ldm.data.occluded_scene import lmdbDataset as ocr_dataset


def mosaic(image, window_size=5, p=None, size=None):
    image = image.resize((128, 32), Image.BICUBIC)
    arr = np.array(image)
    if p is None:
        p = (0, 0)
    if size is None:
        size = arr.shape

    x, y = p
    h = size[0]
    w = size[1]
    ws = window_size
    for i in range(x, min(x + h, arr.shape[0]), ws):
        for j in range(y, min(y + w, arr.shape[1]), ws):
            arr[i:min(i + ws, arr.shape[0]), j:min(j + ws, arr.shape[1]), :] = arr[i, j, :]

    image = Image.fromarray(arr)
    image.save(f'/home/zhouyuxuan/tmp/mosaic{window_size}.jpg')


def fuse(img_HR, img_lr, p=None, size=None):
    img_HR = img_HR.resize((128, 32), Image.BICUBIC)
    img_lr = img_lr.resize((128, 32), Image.BICUBIC)
    arr1 = np.array(img_HR)
    arr2 = np.array(img_lr)

    if p is None:
        p = (0, 0)
    if size is None:
        size = arr1.shape

    x, y = p
    h = min(size[0], arr1.shape[0] - x)
    w = min(size[1], arr2.shape[1] - y)

    arr3 = np.array(arr1)
    arr3[x:x + h, y:y + w, :] = arr2[x:x + h, y:y + w, :]

    arr1 = arr1.astype(np.int32)
    arr2 = arr2.astype(np.int32)
    arr1[x:x + h, y:y + w, :] = (arr1[x:x + h, y:y + w, :] + arr2[x:x + h, y:y + w, :]) // 2
    arr1 = arr1.astype(np.uint8)

    image = Image.fromarray(arr1)
    image.save('/home/zhouyuxuan/tmp/fuse.jpg')

    image = Image.fromarray(arr3)
    image.save('/home/zhouyuxuan/tmp/concat.jpg')

    img_HR.save('/home/zhouyuxuan/tmp/HR.jpg')
    img_lr.save('/home/zhouyuxuan/tmp/lr.jpg')


if __name__ == '__main__':
    tz = textzoom('/home/zhouyuxuan/textzoom/train1')
    img_HR, img_lr, _, _, _, _ = tz.__getitem__(0)
    for i in range(2, 6):
        mosaic(img_HR, window_size=i, size=(32, 64))
    fuse(img_HR, img_lr, p=(0, 64))
