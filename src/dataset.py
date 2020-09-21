import numpy as np
import os
import cv2

from cfg import Configuration as cfg
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_type: str):
        self.dataset_type = dataset_type
        self.data_folder = cfg.training_folder if self.dataset_type == 'training' else cfg.validation_folder

    def __len__(self):
        return len(os.listdir(self.data_folder))

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
        l_image = ciluv_image[..., 0]
        u_image = np.floor((ciluv_image[..., 1] / 256) * 50)
        v_image = np.floor((ciluv_image[..., 2] / 256) * 50)

        return image, l_image, u_image, v_image

    def process_validation(self, index):
        image = cv2.imread(
            os.path.join(self.data_folder,
                         os.listdir(self.data_folder)[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bw_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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
