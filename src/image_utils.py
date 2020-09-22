import torch
import numpy as np
import cv2
import numpy as np
from cfg import Configuration as cfg
import torchvision.transforms as tsfm


def show_validation_summaries(writer, images, bw_images, u_preds, v_preds,
                              step):

    output = luv_to_rgb(bw_images, u_preds, v_preds)[0]
    output_transformed = luv_to_rgb(bw_images, u_preds, v_preds)[1]
    writer.add_images('validation/images', images, step)
    writer.add_images('validation/bw_images', bw_images, step)

    writer.add_images(
        'validation/outputs/luv',
        output,
        step,
    )
    writer.add_images(
        'validation/outputs/rgb',
        output_transformed,
        step,
    )
    writer.flush()


def show_training_summaries(writer, images, l_images, u_images, v_images,
                            u_preds, v_preds, step):

    target = luv_to_rgb(l_images, u_images, v_images)[0]
    output = luv_to_rgb(l_images, u_preds, v_preds)[0]
    target_transformed = luv_to_rgb(l_images, u_images, v_images)[1]
    output_transformed = luv_to_rgb(l_images, u_preds, v_preds)[1]
    writer.add_images('training/images', images, step)

    # writer.add_images('training/input/l', l_image, step)
    # writer.add_images('training/input/u', u_image, step)
    # writer.add_images('training/input/v', v_image, step)

    # writer.add_images('training/output/u', u_preds, step)
    # writer.add_images('training/output/v', v_preds, step)

    # writer.add_images('training/input/R', image[:,0,...].unsqueeze(1), step)
    # writer.add_images('training/input/G', image[:,1,...].unsqueeze(1), step)
    # writer.add_images('training/input/B', image[:,2,...].unsqueeze(1), step)

    writer.add_images(
        'training/targets/luv',
        target,
        step,
    )
    writer.add_images(
        'training/outputs/luv',
        output,
        step,
    )
    writer.add_images(
        'training/targets/rgb',
        target_transformed,
        step,
    )
    writer.add_images(
        'training/outputs/rgb',
        output_transformed,
        step,
    )
    writer.flush()


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


class ToTensor(object):
    def __call__(self, pic):
        return tsfm.functional.to_tensor(pic.copy())


class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, image):
        if torch.rand(1) < self.p:
            return np.flipud(image)
        return image


class RandomVerticalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, image):
        if torch.rand(1) < self.p:
            return np.fliplr(image)
        return image
