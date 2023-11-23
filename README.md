# RGDiffSR
The official pytorch implementation of Paper: RECOGNITION-GUIDED DIFFUSION MODEL FOR SCENE TEXT IMAGE SUPER-RESOLUTION
<p align="center">
<img src=./RGDiffSR.png />
</p>

[paper](http://arxiv.org/abs/2311.13317)

## Installation

Environment preparation: (Python 3.8 + PyTorch 1.7.0 + Torchvision 0.8.1 + pytorch_lightning 1.5.10 + CUDA 11.0)

```
conda create -n RGDiffSR python=3.8
git clone git@github.com:shercoo/RGDiffSR.git
cd RGDiffSR
pip install -r requirements.txt
```

You can also refer to [taming-transformers](https://github.com/CompVis/taming-transformers) for the installation of taming-transformers library (Needed if VQGAN is applied).

## Dataset preparation

Download the TextZoom dataset at [TextZoom](https://github.com/WenjiaWang0312/TextZoom). 

## Model checkpoints

Download the pre-trained recognizers [Aster](https://github.com/ayumiymk/aster.pytorch), [Moran](https://github.com/Canjie-Luo/MORAN_v2), [CRNN](https://github.com/meijieru/crnn.pytorch).

Download the checkpoints of pre-trained VQGAN and RGDiffSR at [Baidu Netdisk](https://pan.baidu.com/s/1SV7GHY0kfHB6s7eC3tUFqA?pwd=yws3). Password: `yws3`

## Training

First train the latent encoder (VQGAN) model.

```shell
CUDA_VISIBLE_DEVICES=<GPU_IDs> python main.py -b configs/autoencoder/vqgan_2x.yaml -t --gpus <GPU_IDS>     
```

Put the pre-trained VQGAN model in `checkpoints/`.

```shell
CUDA_VISIBLE_DEVICES=<GPU_IDs> python main.py -b configs/latent-diffusion/sr_best.yaml -t --gpus <GPU_IDS>
```

## Testing

Put the pre-trained RGDiffSR model in `checkpoints/`.

```shell
CUDA_VISIBLE_DEVICES=<GPU_IDs> python test.py -b configs/latent-diffusion/sr_test.yaml  --gpus <GPU_IDS>
```

You can manually modify the test dataset directory in `sr_test.yaml` for test on different difficulty of TextZoom dataset. 

## License

The model is licensed under the [MIT license](LICENSE).

## Acknowledgement 
Our code is built on the [latent-diffusion](https://github.com/CompVis/latent-diffusion/tree/main) and [TATT](https://github.com/mjq11302010044/TATT) repositories. Thanks to their research!

