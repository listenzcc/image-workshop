#!/usr/bin/python3

import os
import warnings
import argparse
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid, _log_api_usage_once

from dataset import UnpairedDepthDataset
from model import Generator, GlobalGenerator2, InceptionV3
from base_dataset import get_params, get_transform

opt = argparse.Namespace(
    name='opensketch_style',
    checkpoints_dir='checkpoints',
    results_dir='../processed/informative-drawings/image/shared1000',
    geom_name='feats2Geom',
    batchSize=1,
    dataroot='../data/image/shared1000',
    depthroot='',
    input_nc=3,
    output_nc=1,
    geom_nc=3,
    every_feat=1,
    num_classes=55,
    midas=0,
    ngf=64,
    n_blocks=3,
    size=256,
    cuda=True,
    n_cpu=8,
    which_epoch='latest',
    aspect_ratio=1.0,
    mode='test',
    load_size=256,
    crop_size=256,
    max_dataset_size=np.inf,
    preprocess='resize_and_crop',
    no_flip=False,
    norm='instance',
    predict_depth=1,
    save_input=0,
    reconstruct=0,
    how_many=1000)
print('--- opt ---', opt)

opt.no_flip = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available() and not opt.cuda:
    warnings.warn(
        "You have a CUDA device, so you should probably run with --cuda")

with torch.no_grad():
    # Networks
    net_G = 0
    net_G = Generator(opt.input_nc, opt.output_nc, opt.n_blocks)
    net_G.cuda()

    net_GB = 0
    if opt.reconstruct == 1:
        net_GB = Generator(opt.output_nc, opt.input_nc, opt.n_blocks)
        net_GB.cuda()
        net_GB.load_state_dict(torch.load(os.path.join(
            opt.checkpoints_dir, opt.name, 'netG_B_%s.pth' % opt.which_epoch)))
        net_GB.eval()

    netGeom = 0
    if opt.predict_depth == 1:
        usename = opt.name
        if (len(opt.geom_name) > 0) and (os.path.exists(os.path.join(opt.checkpoints_dir, opt.geom_name))):
            usename = opt.geom_name
        myname = os.path.join(opt.checkpoints_dir,
                              'netGeom_%s.pth' % opt.which_epoch)
        netGeom = GlobalGenerator2(
            768, opt.geom_nc, n_downsampling=1, n_UPsampling=3)

        netGeom.load_state_dict(torch.load(myname))
        netGeom.cuda()
        netGeom.eval()

        numclasses = opt.num_classes
        # load pretrained inception
        net_recog = InceptionV3(opt.num_classes, False, use_aux=True,
                                pretrain=True, freeze=True, every_feat=opt.every_feat == 1)
        net_recog.cuda()
        net_recog.eval()

    # Load state dicts
    net_G.load_state_dict(torch.load(os.path.join(
        opt.checkpoints_dir, opt.name, 'netG_A_%s.pth' % opt.which_epoch)))
    print('loaded', os.path.join(opt.checkpoints_dir,
          opt.name, 'netG_A_%s.pth' % opt.which_epoch))

    # Set model's test mode
    net_G.eval()

    def processed_data_to_img(data, **kwargs):
        grid = make_grid(data, **kwargs)
        # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(
            1, 2, 0).to("cpu", torch.uint8).numpy()
        image = Image.fromarray(ndarr)
        return image

    def process_image(image: Image, opt: argparse.Namespace):
        img_r = image.convert('RGB')
        transform_params = get_params(opt, img_r.size)

        A_transform = get_transform(
            opt, transform_params, grayscale=(opt.input_nc == 1), norm=False)
        B_transform = get_transform(
            opt, transform_params, grayscale=(opt.output_nc == 1), norm=False)

        transforms_r = [transforms.Resize(int(opt.size), Image.BICUBIC),
                        transforms.ToTensor()]
        A_transform = transforms.Compose(transforms_r)

        img_r = A_transform(img_r)

        # B_mode = 'L'
        # if opt.output_nc == 3:
        #     B_mode = 'RGB'

        # img_depth = 0
        # label = 0
        # img_path = 0
        # index = 0
        # base = 0

        # input_dict = {'r': img_r, 'depth': img_depth, 'path': img_path,
        #               'index': index, 'name': base, 'label': label}

        input_image = img_r.cuda()
        data = net_G(input_image)
        image = processed_data_to_img(data)

        return image


example_image = Image.open('example.png')
processed_image = process_image(example_image, opt)
processed_image.save('example_processed.png')
print(processed_image)
