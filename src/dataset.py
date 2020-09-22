import torchvision.transforms as tsfm
import numpy as np
import os
import cv2

from cfg import Configuration as cfg
import torch
from image_utils import RandomHorizontalFlip, RandomVerticalFlip, ToTensor


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type
        self.data_folder = cfg.training_folder if self.dataset_type == 'training' else cfg.validation_folder
        self.training_transforms = tsfm.Compose([
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
        ])

    def __len__(self):
        return 2 * cfg.batch_size if cfg.just_testing else len(
            os.listdir(self.data_folder))

    def __getitem__(self, index):
        if self.dataset_type == 'training':
            image, l_image, u_image, v_image = self.process_training(index)
            return image, l_image, u_image, v_image
        else:
            image, bw_image = self.process_validation(index)
            return image, bw_image

    def process_training(self, index):
        image = cv2.imread(
            os.path.join(self.data_folder,
                         os.listdir(self.data_folder)[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ciluv_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        l_image = np.expand_dims(ciluv_image[..., 0], -1)
        u_image = np.floor(
            (ciluv_image[..., 1] / 256) * cfg.bin_no)  #.astype('uint8')
        v_image = np.floor(
            (ciluv_image[..., 2] / 256) * cfg.bin_no)  #.astype('uint8')

        image = self.training_transforms(image)
        l_image = self.training_transforms(l_image)
        u_image = self.training_transforms(u_image)
        v_image = self.training_transforms(v_image)
        return image, l_image, u_image, v_image

    def process_validation(self, index):
        image = cv2.imread(
            os.path.join(self.data_folder,
                         os.listdir(self.data_folder)[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bw_image = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), -1)
        image = tsfm.ToTensor()(image)
        bw_image = tsfm.ToTensor()(bw_image)

        return image, bw_image


def test_dataset():
    train_dataset = Dataset('training')
    image, l_image, u_image, v_image = train_dataset[10]
    print(image.shape)
    print(l_image.shape)
    print(u_image.shape)
    print(v_image.shape)
    print(len(train_dataset))

    validation_dataset = Dataset('validation')
    image, br_image = validation_dataset[10]
    print(image.shape)
    print(br_image.shape)
    print(len(validation_dataset))


def main():
    test_dataset()
    # pass


if __name__ == '__main__':
    main()
