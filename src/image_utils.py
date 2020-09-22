import torch
import cv2
import numpy as np
from cfg import Configuration as cfg
import torchvision.transforms as tsfm


def unbin(image: torch.Tensor):
    tensor = torch.floor(image * 256 / float(cfg.bin_no))
    np_form = tensor.detach().cpu().numpy()
    return tensor, np_form


def luv_to_rgb(image_l: torch.Tensor, image_u: torch.Tensor,
               image_v: torch.Tensor):
    image_l = image_l.detach().cpu()
    image_l = [np.array(tsfm.ToPILImage()(image_li)) for image_li in image_l]
    image_l = np.stack(np.expand_dims(image_l, 1))
    image_u = unbin(image_u)[1]
    image_v = unbin(image_v)[1]

    combined = np.concatenate((image_l, image_u, image_v), axis=1)
    combined = np.transpose(combined, axes=(0, 2, 3, 1))
    transformed = [
        cv2.cvtColor(combined_, cv2.COLOR_LUV2RGB)
        for combined_ in combined.astype('uint8')
    ]
    transformed = np.stack(transformed)
    combined = np.transpose(combined, axes=(0, 3, 1, 2))
    combined /= 255.
    transformed = np.transpose(transformed, axes=(0, 3, 1, 2))

    return combined, transformed
