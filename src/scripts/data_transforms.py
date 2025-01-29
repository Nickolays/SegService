import numpy as np
import torch
from PIL import Image
# from torchvision.transforms.v2 import Compose, Resize, ToTensor  # , Normalize

# from skimage import measure
# from scipy.interpolate import RegularGridInterpolator

import os, cv2
from tqdm import tqdm
# from natsort import natsorted

import warnings
warnings.filterwarnings('ignore')


def prepare_input(img, image_size=(512, 512), device='cpu'):
    """  """
    assert isinstance(img, Image.Image)
    # TODO: Add converter from Image to numpy
    # TODO: Use torch transform for convert to tensor and resize
    if isinstance(img, Image.Image):
        img = np.asarray(img)
    if img.shape[:2] != image_size:
        img = cv2.resize(img, image_size)

    img = img / 255
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    img = img.astype(np.float32)
    img = torch.from_numpy(img).to(device)

    # # Write torch transforms for input image 
    # transforms = Compose([
    #     Resize(image_size),
    #     ToTensor()
    # ])
    # img = transforms(img)
    # img = img.to(device)

    print(img.shape, img.max())

    return img

def prepare_output(pred, threshold=None, device='cpu'):
    """ Convert torch.Tensor to numpy """
    if len(pred) > 1:
        pred = pred[-1]
    if pred.ndim == 4:
        pred = pred[0]

    if device == "cuda":
        pred = pred.cpu().detach().numpy()
    else:
        pred = pred.detach().numpy()

    pred = sigmoid(pred)
    if threshold:
        pred = np.where(pred >= threshold, 1, 0)
    return np.transpose(pred, (1, 2, 0))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# def postprocess(self, predict, threshold=0.5):
#         """ Resized predicted image into the main plot shape """
#         print(f"postprocess input shape:  {predict.shape}")
#         predict = predict[0].detach().numpy()  # shape is torch.Size([1, 1, 512, 512])
#         # Reshape predict 
#         predict = np.permute_dims(predict, (1, 2, 0))
#         # Cut batch size and channels
#         # predict = predict[0]  # (h, w, n_cls)
#         if threshold != 0:
#             predict = np.where(predict >= threshold, 1, 0)
#         print(f"Postprocess output shape:  {predict.shape}")
#         return predict