from torch.utils.tensorboard import SummaryWriter
import shutil
import argparse
import numpy as np
import torch
import torchvision.transforms as tsfm
from tqdm import tqdm

import os
from cfg import Configuration as cfg
from dataset import Dataset
from model import Network
from image_utils import luv_to_rgb, show_training_summaries, show_validation_summaries


class Train:
    def __init__(self, name: str, device=torch.device('cuda')):
        self.name = name
        self.device = device
        self.model = Network().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.writer = SummaryWriter(os.path.join(cfg.work_dir, self.name))

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
            drop_last=True,
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
                L_2 = 0
                for param in self.model.parameters():
                    L_2 += (param**2).sum()
                    loss += 2 / image.size(0) * cfg.REG_COEF * (L_2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if epoch % 10 == 0:
                show_training_summaries(
                    self.writer,
                    image,
                    l_image,
                    u_image,
                    v_image,
                    u_preds,
                    v_preds,
                    epoch,
                )
            self.model.eval()
            validation_loss = 0
            for batch in self.validation_dataloader:

                image, bw_image = batch
                image = image.to(self.device)
                bw_image = bw_image.to(self.device).float()

                u_logits, v_logits = self.model(bw_image)
                u_preds = torch.argmax(torch.softmax(u_logits, 1),
                                       1).unsqueeze(1)
                v_preds = torch.argmax(torch.softmax(v_logits, 1),
                                       1).unsqueeze(1)
                loss = self.criterion(u_logits.flatten(2, 3),
                                      u_image.flatten(2, 3).squeeze())
                loss += self.criterion(v_logits.flatten(2, 3),
                                       v_image.flatten(2, 3).squeeze())
                validation_loss += loss.item()

            if epoch % 10 == 0:
                show_validation_summaries(
                    self.writer,
                    image,
                    bw_image,
                    u_preds,
                    v_preds,
                    epoch,
                )

            self.writer.add_scalars(
                'loss',
                {
                    'training': epoch_loss / len(self.training_dataloader),
                    'validation':
                    validation_loss / len(self.validation_dataloader)
                },
                epoch,
            )
            self.writer.flush()

        if not os.path.exists(os.path.join(cfg.work_dir, 'trained_models')):
            os.mkdir(os.path.join(cfg.work_dir, 'trained_models'))
        torch.save(
            self.model.state_dict(),
            os.path.join(cfg.work_dir, 'trained_models', self.name + '.pt'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'project_name',
        help="name to be shown in tensorboard/under which model will be saved",
    )
    parser.add_argument(
        '--test_session',
        help='test session will load less images',
        action='store_true',
    )
    args = parser.parse_args()
    cfg.just_testing = args.test_session
    if os.path.exists(os.path.join(cfg.work_dir, args.project_name)):
        shutil.rmtree(os.path.join(cfg.work_dir, args.project_name))

    trainer = Train(args.project_name)
    trainer.fit()


if __name__ == '__main__':
    main()
