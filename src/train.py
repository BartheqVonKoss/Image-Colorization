from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torchvision.transforms as tsfm
from tqdm import tqdm

import os
from cfg import Configuration as cfg
from dataset import Dataset
from model import Network
from image_utils import luv_to_rgb
# from utils import TrainingHandler, make_confusion_matrix, make_histogram, loss_hinge, signum, make_barplot


class Train:
    def __init__(self, name: str, device=torch.device('cuda')):
        self.device = device
        self.model = Network().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.writer = SummaryWriter(os.path.join(cfg.work_dir, name))

        self.training_dataset = Dataset('training')
        self.validation_dataset = Dataset('validation')

        self.training_dataloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.validation_dataloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=cfg.batch_size,
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def fit(self):
        self.model.train()
        for epoch in tqdm(range(cfg.EPOCHS)):
            epoch_loss = 0
            for batch in self.training_dataloader:

                image, l_image, u_image, v_image = batch
                image = image.to(self.device)
                l_image = l_image.to(self.device).float()
                u_image = u_image.to(self.device).long()
                v_image = v_image.to(self.device).long()

                u_logits, v_logits = self.model(l_image)
                u_preds = torch.argmax(torch.softmax(u_logits, 1),
                                       1).unsqueeze(1)
                v_preds = torch.argmax(torch.softmax(v_logits, 1),
                                       1).unsqueeze(1)
                loss = self.criterion(u_logits.flatten(2, 3),
                                      u_image.flatten(2, 3).squeeze())
                loss += self.criterion(v_logits.flatten(2, 3),
                                       v_image.flatten(2, 3).squeeze())
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % 10 == 0:
                target = luv_to_rgb(l_image, u_image, v_image)[0]
                output = luv_to_rgb(l_image, u_preds, v_preds)[0]
                target_transformed = luv_to_rgb(l_image, u_image, v_image)[1]
                output_transformed = luv_to_rgb(l_image, u_preds, v_preds)[1]
                self.writer.add_images('training/images', image, epoch)

                # self.writer.add_images('training/input/l', l_image, epoch)
                # self.writer.add_images('training/input/u', u_image, epoch)
                # self.writer.add_images('training/input/v', v_image, epoch)

                # self.writer.add_images('training/output/u', u_preds, epoch)
                # self.writer.add_images('training/output/v', v_preds, epoch)

                # self.writer.add_images('training/input/R', image[:,0,...].unsqueeze(1), epoch)
                # self.writer.add_images('training/input/G', image[:,1,...].unsqueeze(1), epoch)
                # self.writer.add_images('training/input/B', image[:,2,...].unsqueeze(1), epoch)

                self.writer.add_images(
                    'training/targets/luv',
                    target,
                    epoch,
                )
                self.writer.add_images(
                    'training/outputs/luv',
                    output,
                    epoch,
                )
                self.writer.add_images(
                    'training/targets/rgb',
                    target_transformed,
                    epoch,
                )
                self.writer.add_images(
                    'training/outputs/rgb',
                    output_transformed,
                    epoch,
                )


def main():
    trainer = Train('training')
    trainer.fit()


if __name__ == '__main__':
    main()
