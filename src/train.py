from pathlib import Path
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
import shutil
import argparse
import numpy as np
import torch
import torchvision.transforms as tsfm
from tqdm import tqdm

import os
from src.dataset import Dataset
from src.model import Network
from src.utils.image_utils import show_training_summaries, show_validation_summaries


class Train:
    def __init__(
            self,
            config,
            name: str,
            device=torch.device('cuda'),
            model_path: str = None,
    ):

        self.name = name
        self.config = config
        self.device = device
        self.model = Network(config).to(self.device)
        if model_path is not None:
            chckpt = torch.load(model_path)
            self.model.load_state_dict(chckpt)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.config.lr)
        self.writer = SummaryWriter(
            os.path.join(self.config.work_dir, self.name))

        self.training_dataset = Dataset(dataset_type='training', config=config)
        self.validation_dataset = Dataset(dataset_type='validation',
                                          config=config)

        self.training_dataloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.validation_dataloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.config.batch_size,
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def fit(self):
        self.model.train()
        for epoch in tqdm(range(self.config.EPOCHS)):
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
                    loss += 2 / image.size(0) * self.config.REG_COEF * (L_2)
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

        if not os.path.exists(
                os.path.join(self.config.work_dir, 'trained_models')):
            os.mkdir(os.path.join(self.config.work_dir, 'trained_models'))
        torch.save(
            self.model.state_dict(),
            os.path.join(self.config.work_dir, 'trained_models',
                         self.name + '.pt'))

    def __call__(self) -> None:
        try:
            self.fit()
        except KeyboardInterrupt:
            print("Interupting train")
        print("Saving Last model")
        self.save_model()

    def save_model(self):
        # TODO migrate it or change the way of saving in general
        if not os.path.exists(self.config.work_dir):
            os.makedirs(self.config.work_dir)
        torch.save(self.model.state_dict(),
                   f"{Path(self.config.work_dir, self.name)}_model.pt")
