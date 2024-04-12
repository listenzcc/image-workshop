# %%
# Module import
import io
import os
import cv2
import sys
import time
import torch
import base64
import asyncio
import argparse
import warnings
import threading
import websockets
import contextlib

import numpy as np

from PIL import Image
from pathlib import Path
from loguru import logger

# !!! Dangerous operation
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # noqa

# Local import
from operator import getitem
from torchvision.transforms import Compose
from torchvision.transforms import transforms
from utils import ImageandPatchs, ImageDataset, generate_mask, getGF_from_integral, calculate_processing_res, rgb2gray, \
    applyGridpatch
import midas.utils
from midas.models.midas_net import MidasNet
from midas.models.transforms import Resize, NormalizeImage, PrepareForNet
from lib.multi_depth_model_woauxi import RelDepthModel
from lib.net_tools import strip_prefix_if_present
from pix2pix.options.test_options import TestOptions
from pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel


# %%
# OUR

# MIDAS

# AdelaiDepth

# PIX2PIX : MERGE NET

warnings.simplefilter('ignore', np.RankWarning)


# select device
device = torch.device("cuda")

# Global variables
pix2pixmodel = None
midasmodel = None
srlnet = None
leresmodel = None
factor = None
whole_size_threshold = 3000  # R_max from the paper
# Limit for the GPU (NVIDIA RTX 2080), can be adjusted
GPU_threshold = 1600 - 32

# MAIN PART OF OUR METHOD


def load_merge_network():
    # Load merge network
    opt = TestOptions().parse()
    global pix2pixmodel
    pix2pixmodel = Pix2Pix4DepthModel(opt)
    pix2pixmodel.save_dir = './pix2pix/checkpoints/mergemodel'
    pix2pixmodel.load_networks('latest')
    pix2pixmodel.eval()
    return pix2pixmodel


def load_leres_model():
    leres_model_path = "res101.pth"
    checkpoint = torch.load(leres_model_path)
    leresmodel = RelDepthModel(backbone='resnext101')
    leresmodel.load_state_dict(strip_prefix_if_present(checkpoint['depth_model'], "module."),
                               strict=True)
    del checkpoint
    torch.cuda.empty_cache()
    leresmodel.to(device)
    leresmodel.eval()
    return leresmodel


pix2pixmodel = load_merge_network()
leresmodel = load_leres_model()


def cvt_depth_to_cv2(depth, bits=1, colored=False):
    """Convert depth map to cv2 mat

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    # write_pfm(path + ".pfm", depth.astype(np.float32))
    if colored == True:
        # ! Will not happen
        bits = 1

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1
    # if depth_max>max_val:
    #     print('Warning: Depth being clipped')
    #
    # if depth_max - depth_min > np.finfo("float").eps:
    #     out = depth
    #     out [depth > max_val] = max_val
    # else:
    #     out = 0

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    if bits == 1:
        return out.astype('uint8')
    elif bits == 2:
        return out.astype('uint16')


class CompatibleImage(object):
    """Represents an image that is compatible with the monocular depth estimation algorithm.

    The load_from_bytes method of the CompatibleImage class loads the image from encoded bytes.
    The load_from_image method loads the image from a PIL Image object.
    The load_from_cv2 method loads the image from a NumPy array.
    These methods provide different ways to load an image into the CompatibleImage object, depending on the input format.

    The load_from_bytes method of the CompatibleImage class is the first step in the pipeline. It loads the image from encoded bytes.
    The load_from_image method is the second step in the pipeline. It loads the image from a PIL Image object.
    The load_from_cv2 method is the final step in the pipeline. It loads the image from a NumPy array.

    Together, these methods form a pipeline for loading an image into the CompatibleImage object, allowing for flexibility in input formats.

    Attributes:
        name: The name of the image.

    Methods:
        load_from_bytes: Loads the image from encoded bytes.
        load_from_image: Loads the image from a PIL Image object.
        load_from_cv2: Loads the image from a NumPy array.
        rename: Renames the image.

    Raises:
        None.
    """

    name = 'incoming-image'

    def __init__(self):
        pass

    def load_from_bytes(self, encoded: bytes) -> Image:
        """Loads the image from encoded bytes.

        Args:
            encoded: The encoded bytes of the image.

        Returns:
            The loaded image as a PIL Image object.

        Raises:
            None.

        Generate
            self.image
            self.rgb_image
            self.encoded
        """

        img_str = io.BytesIO(base64.b64decode(encoded))
        image = Image.open(img_str).convert('RGB')
        rgb_image = np.array(image) / 255.0
        self.image = image
        self.rgb_image = rgb_image
        self.encoded = encoded
        return image

    def load_from_image(self, image: Image, format: str = 'JPEG') -> bytes:
        """Loads the image from a PIL Image object.

        load_from_image --> load_from_bytes

        Args:
            image: The PIL Image object to load.
            format: The format of the image (default is 'JPEG').

        Returns:
            The encoded bytes of the loaded image.

        Raises:
            None.
        """

        buffered = io.BytesIO()
        image.save(buffered, format=format)
        encoded = base64.b64encode(buffered.getvalue())
        self.load_from_bytes(encoded)
        return encoded

    def load_from_cv2(self, mat: np.ndarray, cvt: int = cv2.COLOR_BGR2RGB) -> Image:
        """Loads the image from a NumPy array.

        load_from_cv2 --> load_from_image --> load_from_bytes

        Args:
            mat: The NumPy array representing the image.
            cvt: The color conversion code (default is cv2.COLOR_BGR2RGB).

        Returns:
            The loaded image as a PIL Image object.

        Raises:
            None.
        """

        if cvt is not None:
            mat = cv2.cvtColor(mat, cvt)
        image = Image.fromarray(mat)
        self.load_from_image(image)
        return image

    def rename(self, name: str) -> str:
        """Renames the image.

        Args:
            name: The new name for the image.

        Returns:
            The new name of the image.

        Raises:
            None.
        """
        self.name = name
        return name


def run(encoded, option):
    """Runs the monocular depth estimation algorithm on an encoded image.

    Args:
        encoded: The encoded image.
        option: The options for the depth estimation algorithm.

    Returns:
        The depth estimation result as an image.

    Raises:
        None.
    """
    # Generate mask used to smoothly blend the local pathc estimations to the base estimate.
    # It is arbitrarily large to avoid artifacts during rescaling for each crop.
    mask_org = generate_mask((3000, 3000))
    mask = mask_org.copy()

    # Value x of R_x defined in the section 5 of the main paper.
    r_threshold_value = 0.2

    # Go through all images in input directory
    print("start processing")

    images = CompatibleImage()
    images.load_from_bytes(encoded)

    print(f'processing image {images.name}')
    # raise Exception('Stop here')

    # Load image from dataset
    img = images.rgb_image
    input_resolution = img.shape

    scale_threshold = 3  # Allows up-scaling with a scale up to 3

    # Find the best input resolution R-x. The resolution search described in section 5-double estimation of the main paper and section B of the
    # supplementary material.
    whole_image_optimal_size, patch_scale = calculate_processing_res(img, option.net_receptive_field_size,
                                                                     r_threshold_value, scale_threshold,
                                                                     whole_size_threshold)

    print('\t wholeImage being processed in :', whole_image_optimal_size)

    # Generate the base estimate using the double estimation.
    whole_estimate = doubleestimate(img, option.net_receptive_field_size, whole_image_optimal_size,
                                    option.pix2pixsize, option.depthNet)

    # Compute the multiplier described in section 6 of the main paper to make sure our initial patch can select
    # small high-density regions of the image.
    global factor
    factor = max(min(1, 4 * patch_scale *
                     whole_image_optimal_size / whole_size_threshold), 0.2)
    print('Adjust factor is:', 1/factor)

    # Compute the default target resolution.
    if img.shape[0] > img.shape[1]:
        a = 2 * whole_image_optimal_size
        b = round(2 * whole_image_optimal_size *
                  img.shape[1] / img.shape[0])
    else:
        a = round(2 * whole_image_optimal_size *
                  img.shape[0] / img.shape[1])
        b = 2 * whole_image_optimal_size
    b = int(round(b / factor))
    a = int(round(a / factor))

    # recompute a, b and saturate to max res.
    if max(a, b) > option.max_res:
        print('Default Res is higher than max-res: Reducing final resolution')
        if img.shape[0] > img.shape[1]:
            a = option.max_res
            b = round(option.max_res * img.shape[1] / img.shape[0])
        else:
            a = round(option.max_res * img.shape[0] / img.shape[1])
            b = option.max_res
        b = int(b)
        a = int(a)

    img = cv2.resize(img, (b, a), interpolation=cv2.INTER_CUBIC)

    # Extract selected patches for local refinement
    base_size = option.net_receptive_field_size*2
    patchset = generate_patches(img, base_size)

    print('Target resolution: ', img.shape)

    # Computing a scale in case user prompted to generate the results as the same resolution of the input.
    # Notice that our method output resolution is independent of the input resolution and this parameter will only
    # enable a scaling operation during the local patch merge implementation to generate results with the same resolution
    # as the input.
    if option.output_resolution == 1:
        mergein_scale = input_resolution[0] / img.shape[0]
        print('Dynamicly change merged-in resolution; scale:', mergein_scale)
    else:
        mergein_scale = 1

    imageandpatchs = ImageandPatchs(
        None, None, patchset, img, mergein_scale)
    whole_estimate_resized = cv2.resize(whole_estimate, (round(img.shape[1]*mergein_scale),
                                        round(img.shape[0]*mergein_scale)), interpolation=cv2.INTER_CUBIC)
    imageandpatchs.set_base_estimate(whole_estimate_resized.copy())
    imageandpatchs.set_updated_estimate(whole_estimate_resized.copy())

    print('\t Resulted depthmap res will be :',
          whole_estimate_resized.shape[:2])
    print(f'patchs to process: {len(imageandpatchs)}')

    # Enumerate through all patches, generate their estimations and refining the base estimate.
    for patch_ind in range(len(imageandpatchs)):

        # Get patch information
        patch = imageandpatchs[patch_ind]  # patch object
        patch_rgb = patch['patch_rgb']  # rgb patch
        # corresponding patch from base
        patch_whole_estimate_base = patch['patch_whole_estimate_base']
        rect = patch['rect']  # patch size and location
        patch_id = patch['id']  # patch ID
        # the original size from the unscaled input
        org_size = patch_whole_estimate_base.shape
        print('\t processing patch', patch_ind, '|', rect)

        # We apply double estimation for patches. The high resolution value is fixed to twice the receptive
        # field size of the network for patches to accelerate the process.
        patch_estimation = doubleestimate(patch_rgb, option.net_receptive_field_size, option.patch_netsize,
                                          option.pix2pixsize, option.depthNet)

        patch_estimation = cv2.resize(patch_estimation, (option.pix2pixsize, option.pix2pixsize),
                                      interpolation=cv2.INTER_CUBIC)

        patch_whole_estimate_base = cv2.resize(patch_whole_estimate_base, (option.pix2pixsize, option.pix2pixsize),
                                               interpolation=cv2.INTER_CUBIC)

        # Merging the patch estimation into the base estimate using our merge network:
        # We feed the patch estimation and the same region from the updated base estimate to the merge network
        # to generate the target estimate for the corresponding region.
        pix2pixmodel.set_input(patch_whole_estimate_base, patch_estimation)

        # Run merging network
        pix2pixmodel.test()
        visuals = pix2pixmodel.get_current_visuals()

        prediction_mapped = visuals['fake_B']
        prediction_mapped = (prediction_mapped+1)/2
        prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

        mapped = prediction_mapped

        # We use a simple linear polynomial to make sure the result of the merge network would match the values of
        # base estimate
        p_coef = np.polyfit(mapped.reshape(-1),
                            patch_whole_estimate_base.reshape(-1), deg=1)
        merged = np.polyval(p_coef, mapped.reshape(-1)
                            ).reshape(mapped.shape)

        merged = cv2.resize(
            merged, (org_size[1], org_size[0]), interpolation=cv2.INTER_CUBIC)

        # Get patch size and location
        w1 = rect[0]
        h1 = rect[1]
        w2 = w1 + rect[2]
        h2 = h1 + rect[3]

        # To speed up the implementation, we only generate the Gaussian mask once with a sufficiently large size
        # and resize it to our needed size while merging the patches.
        if mask.shape != org_size:
            mask = cv2.resize(
                mask_org, (org_size[1], org_size[0]), interpolation=cv2.INTER_LINEAR)

        tobemergedto = imageandpatchs.estimation_updated_image

        # Update the whole estimation:
        # We use a simple Gaussian mask to blend the merged patch region with the base estimate to ensure seamless
        # blending at the boundaries of the patch region.
        tobemergedto[h1:h2, w1:w2] = np.multiply(
            tobemergedto[h1:h2, w1:w2], 1 - mask) + np.multiply(merged, mask)
        imageandpatchs.set_updated_estimate(tobemergedto)

    # Output the result
    res_image = CompatibleImage()
    mat = cvt_depth_to_cv2(cv2.resize(imageandpatchs.estimation_updated_image,
                                      (input_resolution[1],
                                       input_resolution[0]),
                                      interpolation=cv2.INTER_CUBIC))
    print(mat)
    res_image.load_from_cv2(mat)

    print("finished")
    return res_image

# Generating local patches to perform the local refinement described in section 6 of the main paper.


def generate_patches(img, base_size):

    # Compute the gradients as a proxy of the contextual cues.
    img_gray = rgb2gray(img)
    whole_grad = np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)) +\
        np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3))

    threshold = whole_grad[whole_grad > 0].mean()
    whole_grad[whole_grad < threshold] = 0

    # We use the integral image to speed-up the evaluation of the amount of gradients for each patch.
    gf = whole_grad.sum()/len(whole_grad.reshape(-1))
    grad_integral_image = cv2.integral(whole_grad)

    # Variables are selected such that the initial patch size would be the receptive field size
    # and the stride is set to 1/3 of the receptive field size.
    blsize = int(round(base_size/2))
    stride = int(round(blsize*0.75))

    # Get initial Grid
    patch_bound_list = applyGridpatch(blsize, stride, img, [0, 0, 0, 0])

    # Refine initial Grid of patches by discarding the flat (in terms of gradients of the rgb image) ones. Refine
    # each patch size to ensure that there will be enough depth cues for the network to generate a consistent depth map.
    print("Selecting patchs ...")
    patch_bound_list = adaptive_selection(
        grad_integral_image, patch_bound_list, gf)

    return sorted(
        patch_bound_list.items(),
        key=lambda x: getitem(x[1], 'size'),
        reverse=True,
    )


# Adaptively select patches
def adaptive_selection(integral_grad, patch_bound_list, gf):
    patch_list = {}
    count = 0
    height, width = integral_grad.shape

    search_step = int(32/factor)

    # Go through all patches
    for c in range(len(patch_bound_list)):
        # Get patch
        bbox = patch_bound_list[str(c)]['rect']

        # Compute the amount of gradients present in the patch from the integral image.
        cgf = getGF_from_integral(integral_grad, bbox)/(bbox[2]*bbox[3])

        # Check if patching is beneficial by comparing the gradient density of the patch to
        # the gradient density of the whole image
        if cgf >= gf:
            bbox_test = bbox.copy()
            patch_list[str(count)] = {}

            # Enlarge each patch until the gradient density of the patch is equal
            # to the whole image gradient density
            while True:

                bbox_test[0] = bbox_test[0] - search_step // 2
                bbox_test[1] = bbox_test[1] - search_step // 2

                bbox_test[2] = bbox_test[2] + search_step
                bbox_test[3] = bbox_test[3] + search_step

                # Check if we are still within the image
                if bbox_test[0] < 0 or bbox_test[1] < 0 or bbox_test[1] + bbox_test[3] >= height \
                        or bbox_test[0] + bbox_test[2] >= width:
                    break

                # Compare gradient density
                cgf = getGF_from_integral(
                    integral_grad, bbox_test)/(bbox_test[2]*bbox_test[3])
                if cgf < gf:
                    break
                bbox = bbox_test.copy()

            # Add patch to selected patches
            patch_list[str(count)]['rect'] = bbox
            patch_list[str(count)]['size'] = bbox[2]
            count = count + 1

    # Return selected patches
    return patch_list


# Generate a double-input depth estimation
def doubleestimate(img, size1, size2, pix2pixsize, net_type):
    # Generate the low resolution estimation
    estimate1 = singleestimate(img, size1, net_type)
    # Resize to the inference size of merge network.
    estimate1 = cv2.resize(
        estimate1, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Generate the high resolution estimation
    estimate2 = singleestimate(img, size2, net_type)
    # Resize to the inference size of merge network.
    estimate2 = cv2.resize(
        estimate2, (pix2pixsize, pix2pixsize), interpolation=cv2.INTER_CUBIC)

    # Inference on the merge model
    pix2pixmodel.set_input(estimate1, estimate2)
    pix2pixmodel.test()
    visuals = pix2pixmodel.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped+1)/2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
        torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped


# Generate a single-input depth estimation
def singleestimate(img, msize, net_type):
    if msize > GPU_threshold:
        print(" \t \t DEBUG| GPU THRESHOLD REACHED",
              msize, '--->', GPU_threshold)
        msize = GPU_threshold

    if net_type == 0:
        return estimatemidas(img, msize)
    elif net_type == 1:
        return estimatesrl(img, msize)
    elif net_type == 2:
        return estimateleres(img, msize)


# Inference on SGRNet
def estimatesrl(img, msize):
    # SGRNet forward pass script adapted from https://github.com/KexianHust/Structure-Guided-Ranking-Loss
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_resized = cv2.resize(
        img, (msize, msize), interpolation=cv2.INTER_CUBIC).astype('float32')
    tensor_img = img_transform(img_resized)

    # Forward pass
    input_img = torch.autograd.Variable(
        tensor_img.cuda().unsqueeze(0), volatile=True)
    with torch.no_grad():
        output = srlnet(input_img)

    # Normalization
    depth = output.squeeze().cpu().data.numpy()
    min_d, max_d = depth.min(), depth.max()
    depth_norm = (depth - min_d) / (max_d - min_d)

    depth_norm = cv2.resize(
        depth_norm, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    return depth_norm

# Inference on MiDas-v2


def estimatemidas(img, msize):
    # MiDas -v2 forward pass script adapted from https://github.com/intel-isl/MiDaS/tree/v2

    transform = Compose(
        [
            Resize(
                msize,
                msize,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    img_input = transform({"image": img})["image"]

    # Forward pass
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = midasmodel.forward(sample)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(
        prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Normalization
    depth_min = prediction.min()
    depth_max = prediction.max()

    if depth_max - depth_min > np.finfo("float").eps:
        prediction = (prediction - depth_min) / (depth_max - depth_min)
    else:
        prediction = 0

    return prediction


def scale_torch(img):
    """
    Scale the image and output it in torch.tensor.
    :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]
    if img.shape[2] == 3:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        img = transform(img.astype(np.float32))
    else:
        img = img.astype(np.float32)
        img = torch.from_numpy(img)
    return img

# Inference on LeRes


def estimateleres(img, msize):
    # LeReS forward pass script adapted from https://github.com/aim-uofa/AdelaiDepth/tree/main/LeReS

    rgb_c = img[:, :, ::-1].copy()
    A_resize = cv2.resize(rgb_c, (msize, msize))
    img_torch = scale_torch(A_resize)[None, :, :, :]

    # Forward pass
    with torch.no_grad():
        prediction = leresmodel.inference(img_torch)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(
        prediction, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    return prediction


# Adding necessary input arguments
# ! The default options are useful values, do not change it at all.
option = argparse.Namespace(
    data_dir=None,
    output_dir=None,
    savepatchs=0,
    savewholeest=0,
    output_resolution=1,
    net_receptive_field_size=448,
    pix2pixsize=1024,
    depthNet=2,
    colorize_results=False,
    R0=False,
    R20=False,
    Final=True,
    max_res=np.inf,
    patch_netsize=896)

# %%
# --------------------------------------------------------------------------------
# After everything is OK, start the websocket server
logger.add('log/BoostingMonocularDepth-websocket-server.log', rotation='5MB')


def check_everything_is_ok():
    # Run at the example image, check if everything is OK

    # 1. Check options
    logger.info(f'Using option: {option}')
    logger.info(f'The midas OK, {midas}')
    logger.info(f"Using device: {device}")

    # 2. Load the example image
    path = Path(__file__).parent.joinpath('example.jpg')
    image = Image.open(path)
    inp = CompatibleImage()
    inp.load_from_image(image)
    logger.debug(f'Generated {inp} from image: {path}')

    # 3. Run at the example image
    logger.debug(f'Using example inp: {inp}')
    res = run(inp.encoded, option)
    logger.debug(f'Passed example image check. The res.image: {res.image}')

    # 4. Say hi
    logger.info('---- BoostingMonocularDepth is running ----')
    return


check_everything_is_ok()


class TaskManager(object):
    uid = 0
    tasks = {}
    rlock = threading.RLock()

    def __init__(self):
        pass

    @contextlib.contextmanager
    def lock(self):
        try:
            yield self.rlock.acquire()
        finally:
            self.rlock.release()

    def new_task(self, description: str = ''):
        '''
        Safely create the new task
        '''
        with self.lock():
            uid = self.uid
            self.uid += 1

            dct = dict(
                tic=time.time(),
                uid=uid,
                state='running',
                description=description
            )
            self.tasks[uid] = dct
            logger.debug(f'Created new task: {dct}')
            return uid

    def task_finished(self, uid, state: str = 'finished'):
        '''
        The task of uid finished.
        Record its running times.
        '''
        with self.lock():
            if uid not in self.tasks:
                logger.warning(f'Invalid task ID: {uid}')
                return None

            dct = self.tasks[uid]
            dct['state'] = state
            dct['toc'] = time.time()
            dct['costs'] = dct['toc'] - dct['tic']
            logger.debug(f'Task finished: {dct}')

            return dct


tm = TaskManager()


async def _handler(websocket: websockets.ServerProtocol):
    # recv is the bytes of an image
    recv = await websocket.recv()
    logger.info(f'Received new request: {recv[:80]}...')

    # Wrap the task with tm: the task manager
    uid = tm.new_task()
    try:
        res = run(recv, option)
        tm.task_finished(uid, state='finished')
    except Exception as err:
        tm.task_finished(uid, state=f'failed: {err}')
        logger.error(f'Failed to run task: {err}')

    # Response
    # Fetch the encoded, it is the bytes of the processed image
    resp = res.encoded
    await websocket.send(resp)


async def serve_forever():
    '''
    It seems the queued requests are processed one-by-one.
    '''
    host = 'localhost'
    port = 23401
    async with websockets.serve(_handler, host, port):
        logger.info(f'Waiting on {host}:{port}...')
        await asyncio.Future()


def main():
    asyncio.run(serve_forever())


if __name__ == "__main__":
    main()
    sys.exit(0)
