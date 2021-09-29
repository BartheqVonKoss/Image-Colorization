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
    ):
        self.name = name
        self.config = config
        self.device = device
        self.model = Network(config).to(self.device)
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
        """ Wrapper to train the network,
        catches ctrl-c if train is stopped in the middle of the
        execution"""
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
        print(self.name)
        if self.name is not None:
            torch.save(self.model.state_dict(),
                        f"{Path(self.config.work_dir, self.name)}_model.pt")
            # else:
            #     torch.save(self.model.state_dict(),
            #                os.path.join(self.config.work_dir, 'model.pt'))
    def train_epoch(self):
        """Perform train epoch """

        self.model.train()
        print("Train Epoch:")
        return self._epoch(train=True, dataloader=self.train_dataloader)

    def eval_epoch(self):
        """Perform evaluation epoch """
        self.model.eval()
        with torch.no_grad():
            print("Eval Epoch:")
            return self._epoch(train=False, dataloader=self.eval_dataloader)

    def test_epoch(self) -> Dict:
        """Perform evaluation epoch """
        self.model.eval()
        all_outputs = None
        all_imgs = None
        all_bbox = None
        all_codes = []
        print("Test Epoch:")
        with torch.no_grad():
            for _, batch in enumerate(self.test_dataloader, start=1):
                print(f"{_} / {len(self.test_dataloader)}", end='\r')
                images = batch['img'].to(self.device)
                barcode_boxes = batch["bbox"].to(self.device)
                all_codes += batch["code"]

                outputs = self.model(images)
                if all_outputs is None:
                    all_outputs = outputs
                    all_imgs = images
                    all_bbox = barcode_boxes
                else:
                    all_outputs = torch.cat((all_outputs, outputs), 0)
                    all_imgs = torch.cat((all_imgs, images), 0)
                    all_bbox = torch.cat((all_bbox, barcode_boxes), 0)

                # break

        correct_reconstructions, correct_reconstructions_crop, total_decoded = decode_outputs(
            all_outputs, targets=all_codes, crop=all_bbox)

        return {
            "images": all_imgs,
            "outputs": all_outputs,
            "correct_reconstructions": correct_reconstructions,
            "correct_reconstructions_crop": correct_reconstructions_crop,
            "total_decoded": total_decoded,
            "dataloader": self.test_dataloader,
        }

    def _epoch(self, train: bool,
               dataloader: torch.utils.data.dataloader.DataLoader) -> Dict:
        """Perform an epoch (training or evaluation run over dataloader)

        Args:
        train: bool
            if train the gradients will be backpropagated
        dataloader: torch.utils.data.DataLoader
            dataloader to run calculation over
        """
        total_loss = 0
        all_outputs = None
        all_imgs = None
        all_targets = None
        for _, batch in enumerate(dataloader):
            print(f"{_} / {len(dataloader)}", end='\r')
            images, targets = batch['img'].to(
                self.device), batch['org_img'].to(self.device)
            outputs = self.model(images)
            # targets are in a binned form [0, 1, 2, 3]
            targets = (targets / 255).float()
            # print(targets[0, :10])
            # print(targets.squeeze().shape)
            # print(outputs.shape)
            # print(torch.unique(targets, return_counts=True))

            # print(torch.softmax(outputs, 1).shape)

            # print(torch.argmax(torch.softmax(outputs, 1), 1).shape)
            # print(torch.unique(torch.argmax(torch.softmax(outputs, 1), 1), return_counts=True))
            # # print(outputs.type)

            loss = 1000 * self.loss_fn(outputs.squeeze(), targets.squeeze())
            print(loss)
            # loss = self.loss_fn(outputs, targets)
            total_loss += loss.item()
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if all_outputs is None:
                all_outputs = outputs
                all_imgs = images
                all_targets = targets
            elif len(all_outputs) < 500 and np.random.uniform() > min(
                    max(0.1,
                        len(all_outputs) / 500), 0.9):
                all_outputs = torch.cat((all_outputs, outputs), 0)
                all_imgs = torch.cat((all_imgs, images), 0)
                all_targets = torch.cat((all_targets, targets), 0)
            # break

        correct_reconstructions, _, _ = decode_outputs(all_outputs,
                                                       targets=all_targets)

        return {
            "loss": total_loss,
            "images": all_imgs,
            "targets": all_targets,
            "outputs": all_outputs,
            "correct_reconstructions": correct_reconstructions,
            "dataloader": dataloader,
        }

    def _loop(self) -> None:
        """ perform training """
        # have a graph written to summaries
        self.writer.add_graph(
            self.model,
            next(iter(self.train_dataloader))['img'].to(self.device),
        )
        if self.config.work_dir:
            if not os.path.exists(self.config.work_dir):
                os.makedirs(self.config.work_dir)

        for epoch in range(self.start_epoch, self.max_epochs):
            self.last_epoch = epoch
            print(f"\nEpoch {epoch}/{self.max_epochs}")

            epoch_start_time = time.time()
            train_metrics = self.train_epoch()
            eval_metrics = self.eval_epoch()
            test_metrics = self.test_epoch()
            torch.save(
                {
                    'state_dict':
                    self.model.state_dict(),
                    'epoch':
                    self.last_epoch,
                    'test_reconstruction':
                    test_metrics['correct_reconstructions'],
                    'train_reconstruction':
                    train_metrics['correct_reconstructions'],
                    'eval_reconstruction':
                    eval_metrics['correct_reconstructions'],
                    'optimizer':
                    self.optimizer.state_dict(),
                },
                os.path.join(
                    self.config.work_dir,
                    #  f"model_{self.last_epoch}.pt"))
                    f"model.pt"))
            print(f"\n\t train_loss: {train_metrics['loss']:.4f}, \
                           eval_loss: {eval_metrics['loss']:.4f}")
            if self.summaries:
                status = [
                    self.writer, self.train_dataloader, self.eval_dataloader,
                    self.test_dataloader, self.last_epoch
                ]
                write_summaries(status, train_metrics, eval_metrics,
                                test_metrics)
            if self.scheduler is not None:
                self.writer.add_scalar("Learning Rate",
                                       self.scheduler.get_last_lr()[0], epoch)
                self.scheduler.step()

            # use if you don't want to have all the models saved but just a few
            # self.patience_cnt = self.update_progress(
            # [self.val_losses, self.patience, self.patience_cnt],eval_metrics['loss'])
            # if len(self.val_losses) <= 5:
            #     self.val_losses = np.append(self.val_losses,
            #                                 eval_metrics['loss'])
            # else:
            #     self.val_losses = np.roll(self.val_losses, 1)
            #     self.val_losses[-1] = eval_metrics['loss']
            print(
                f"\t epoch finished in {(time.time() - epoch_start_time):.2f}")

        print("Finished Training")
