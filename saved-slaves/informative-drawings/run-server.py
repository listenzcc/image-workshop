# %%
# Module import
import io
import os
import sys
import torch
import base64
import asyncio
import warnings
import argparse
import websockets
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from pathlib import Path
from torchvision.utils import make_grid

from loguru import logger

# Local import
from model import Generator, GlobalGenerator2, InceptionV3
from base_dataset import get_params, get_transform

sys.path.append(Path(__file__).parent.parent.parent.as_posix())  # noqa
from slaves.util.task_manager import TaskManager

# --------------------
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
        return Image.fromarray(ndarr)

    def process_image(image: Image, opt: argparse.Namespace):
        image_size = image.size

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

        input_image = img_r.cuda()
        data = net_G(input_image)

        image = processed_data_to_img(data)

        image = image.resize(image_size)

        return image


# %%
# --------------------------------------------------------------------------------
# After everything is OK, start the websocket server
logger.add('log/informative-drawing-websocket-server.log', rotation='5MB')
tm = TaskManager(logger)


def check_everything_is_ok():
    # Run at the example image, check if everything is OK

    # 1. Check options
    logger.info(f'Using option: {opt}')
    logger.info(f"Using device: {device}")

    # 2. Load the example image
    path = Path(__file__).parent.joinpath('example.png')
    image = Image.open(path)
    logger.debug(f'Loaded example image {image} from path: {path}')

    # 3. Run at the example image
    processed_image = process_image(image, opt)
    processed_image.save('example_processed.png')
    logger.debug(
        f'Passed example image check. The processed image: {processed_image}')

    # 4. Say hi
    logger.info('---- informative-drawing backend is OK ----')
    return


check_everything_is_ok()


async def _handler(websocket: websockets.ServerProtocol):
    # recv is the bytes of an image
    recv = await websocket.recv()
    logger.info(f'Received new request: {recv[:80]}...')

    # Wrap the task with tm: the task manager
    uid = tm.new_task()
    try:
        # Suppose recv is the bytes of an image
        img_str = io.BytesIO(base64.b64decode(recv))
        image = Image.open(img_str).convert('RGB')
        processed_image = process_image(image, opt)
        tm.task_finished(uid, state='finished')
    except Exception as err:
        tm.task_finished(uid, state=f'failed: {err}')
        logger.error(f'Failed to run task: {err}')

    # Response
    # Fetch the encoded, it is the bytes of the processed image
    buffered = io.BytesIO()
    processed_image.save(buffered, format='JPEG')
    encoded = base64.b64encode(buffered.getvalue())
    resp = encoded
    await websocket.send(resp)


async def serve_forever():
    '''
    It seems the queued requests are processed one-by-one.
    '''
    host = 'localhost'
    port = 23402
    async with websockets.serve(_handler, host, port, max_size=None):
        logger.info(f'Waiting on {host}:{port}...')
        await asyncio.Future()


def main():
    asyncio.run(serve_forever())


if __name__ == "__main__":
    main()
    sys.exit(0)
