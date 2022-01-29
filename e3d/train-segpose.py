import argparse
import os
import logging
from collections import defaultdict
import pdb
from pathlib import Path

from datasets import EventSegHDF5
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
import torch.nn as nn
import numpy as np
import torch
from losses import DiceCoeffLoss, IOULoss
#from mesh_reconstruction.model import MeshDeformationModel
from PIL import Image
from segpose import UNet, UNetDynamic, SegPoseNet
from torch import optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.params import Params
#from segpose.criterion import PoseCriterion, PoseCriterionRel

'''
from pytorch3d.renderer import PerspectiveCameras
from utils.manager import RenderManager
import json
from utils.visualization import plot_camera_scene
'''

class Trainer():

    def __init__(self, 
                    model, 
                    params, 
                    dataset_path="/Datasets/cwang/cvpr2022/v2_animals/combined",
                    device="cuda"):
        self.width = 640
        self.height = 480
        self.path = dataset_path
        self.params = params
        self.model = model
        self.device = device
        self.build_logger()
        self.train_loader, self.val_loader = self.prepare_data()
        self.unet_optimizer, self.pose_optimizer = self.build_optim()
        self.global_step = 0
        self.fine_tuning = params.fine_tuning
        return

    def step(self, data, idx, mode="train"):
        # Criterions
        unet_criterion = nn.BCEWithLogitsLoss()
        '''
        pose_criterion = PoseCriterionRel(
            sax=0.0, saq=self.params.beta, srx=0.0, srq=self.params.gamma
        ).to(self.device)
        '''

        ev_frame, mask_gt, R_gt, T_gt, pose_gt = data

        ev_frame = data['event_frame']
        mask_gt = data['mask']
        R_gt = data['gt_R']
        T_gt = data['gt_T']
        pose_gt = data['gt_pose']

        event_frame = Variable(ev_frame).to(self.device)
        mask_gt = Variable(mask_gt).to(self.device)
        pose_gt = Variable(pose_gt).to(self.device)

        # Casting variables to float
        ev_frame = event_frame.to(device=device, dtype=torch.float)
        mask_gt = mask_gt.to(device=device, dtype=torch.float)

        mask_pred, pose_pred = self.model(ev_frame)

        '''
        if params.fine_tuning and step % (
            train_size // (5 * params.unet_batch_size)
        ) in [2, 3]:
            prev["R_gt"].append(R_gt)
            prev["T_gt"].append(T_gt)
            prev["mask_pred"].append(mask_pred)

            # Fine-tuning through Differentiable Renderer
            if step % (train_size // (5 * params.unet_batch_size)) == 3:
                fine_tuning = True
                # print(step)
                # Concatenate results from previous step
                R_gt = torch.cat(prev["R_gt"])
                T_gt = torch.cat(prev["T_gt"])
                mask_pred_m = torch.cat(prev["mask_pred"])

                mask_pred_m = (
                    torch.sigmoid(mask_pred).squeeze() > params.threshold_conf
                ).type(torch.uint8) * 255
                writer.add_images(
                    "mask-pred-input", mask_pred_m.unsqueeze(1), self.global_step
                )
                # R_gt_m = R_gt.view(-1, *R_gt.size()[2:]).unsqueeze(1)
                # T_gt_m = T_gt.view(-1, *T_gt.size()[2:]).unsqueeze(1)
                # logging.info(
                #     f"unet output shape: {mask_pred_m.shape}, R shape {R_gt.shape}"
                # )
                mesh_model = MeshDeformationModel(self.device, params)
                mesh_model = nn.DataParallel(mesh_model).to(self.device)
                mesh_losses = mesh_model.module.run_optimization(
                    mask_pred_m, R_gt, T_gt, writer, step=self.global_step
                )
                renders = mesh_model.module.render_final_mesh(
                    (R_gt, T_gt), "predict", mask_pred_m.shape[-2:]
                )
                mask_pred_m = renders["silhouettes"].to(self.device)
                image_pred = renders["images"].to(self.device)

                # logging.info(f"mesh defo shape: {image_pred.shape}")
                writer.add_images("masks-pred-mesh-deform", mask_pred_m, self.global_step)
                writer.add_images(
                    "images-pred-mesh-deform",
                    image_pred.permute(0, 3, 1, 2),
                    self.global_step,
                )
                # Cut out batch_size from mask_pred for calculating loss
                mask_pred_m = mask_pred_m[: ev_frame.shape[0]].requires_grad_()

                prev = defaultdict(list)
        '''

        mask_gt = mask_gt.view(-1, *mask_gt.size()[2:]).unsqueeze(1)

        # Compute losses
        unet_loss = unet_criterion(mask_pred, mask_gt)
        '''
        if fine_tuning and mask_pred_m is not None:
            unet_loss += IOULoss().forward(mask_pred_m, mask_gt) * 1.2
            fine_tuning = False
        pose_loss = pose_criterion(pose_pred, pose_gt)
        loss = pose_loss + unet_loss
        '''
        #TODO: get pose loss back
        loss = unet_loss
        self.summary_writer.add_scalar("UNetLoss/Train", unet_loss.item(), self.global_step)
        #self.log("PoseLoss/Train", pose_loss.item(), step)
        self.summary_writer.add_scalar("CompleteLoss/Train:", loss.item(), self.global_step)
        return loss

    def train(self):
        val_losses = []
        step = 0
        mask_pred_m = None
        prev = defaultdict(list)

        self.model.train()

        for epoch in range(100):
            epoch_loss = 0.0
            for i, data in enumerate(self.train_loader):
                loss = self.step(data, i, mode='train')
                # backprop and optimize
                self.unet_optimizer.zero_grad()
                #pose_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(self.model.module.unet.parameters(), 0.1)
                self.unet_optimizer.step()
                #pose_optimizer.step()
                epoch_loss += loss
                print("Loss: {}\r".format(loss.item()))

                self.global_step += 1

        model_dir = os.path.join(params.exper_dir,
                                 f"{params.name}_epochs{params.unet_epochs}_batch{params.unet_batch_size}_minibatch{params.unet_mini_batch_size}.cpt")
        torch.save(
            {
                "model": model.module.state_dict(),
                "unet_optimizer": unet_optimizer.state_dict(),
                "pose_optimizer": pose_optimizer.state_dict(),
            },
            model_dir,
        )
        return 
    
    def generate_paths(self):
        all_h5_files_train = []
        all_h5_files_test = []

        with open(os.path.join(self.path, "train_split.txt"), 'r') as train_file:
            for f in train_file.readlines():
                f = f.strip("\n")
                path = Path(self.path)
                if "h5" in f:
                    all_h5_files_train.append(os.path.join(self.path, f))

        with open(os.path.join(self.path, "test_split.txt"), 'r') as test_file:
            for f in test_file.readlines():
                f = f.strip("\n")
                path = Path(self.path)
                if "h5" in f:
                    all_h5_files_test.append(os.path.join(self.path, f))
        return all_h5_files_train, all_h5_files_test

    def prepare_data(self):
        all_sequences_train, all_sequences_val = self.generate_paths()
        all_datasets_train = []
        all_datasets_val = []
        data_class = EventSegHDF5
        '''
        for i in range(len(all_sequences_train)):
            dataset_one = data_class(all_sequences_train[i],
                                    width=self.width,
                                    height=self.height,
                                    max_length=-1)
            all_datasets_train.append(dataset_one)
        for i in range(len(all_sequences_val)):
            dataset_one = data_class(all_sequences_val[i],
                                    width=self.width,
                                    height=self.height,
                                    max_length=-1)
            all_datasets_val.append(dataset_one)
        '''
        for i in range(2):
            dataset_one = data_class(all_sequences_train[i],
                                    width=self.width,
                                    height=self.height,
                                    max_length=-1)
            all_datasets_train.append(dataset_one)
        for i in range(2):
            dataset_one = data_class(all_sequences_val[i],
                                    width=self.width,
                                    height=self.height,
                                    max_length=-1)
            all_datasets_val.append(dataset_one)

        self.train_dataset = ConcatDataset(all_datasets_train)
        self.val_dataset = ConcatDataset(all_datasets_val)

        # Train, Val split according to datasets
        '''
        train_sampler = ConcatDataSampler(
            train_dataset, batch_size=self.params.unet_batch_size, shuffle=True
        )
        val_sampler = ConcatDataSampler(
            val_dataset, batch_size=self.params.unet_batch_size, shuffle=True
        )
        '''

        train_loader = DataLoader(self.train_dataset, num_workers=0, shuffle=True)
        val_loader = DataLoader(self.val_dataset, num_workers=4, shuffle=False)
        return train_loader, val_loader

    def build_logger(self):
        log_dir = os.path.join(self.params.exper_dir,
        f"runs/{params.name}_LR_{params.unet_learning_rate}_EPOCHS_{params.unet_epochs}_BS_{params.unet_batch_size}")
        self.summary_writer = SummaryWriter(
            log_dir=log_dir
        )

    def build_optim(self):
        unet_params = self.model.module.unet.parameters()
        unet_optimizer = self.params.unet_optimizer(
            unet_params,
            lr=self.params.unet_learning_rate,
            weight_decay=self.params.unet_weight_decay,
            momentum=self.params.unet_momentum,
        )

        '''
        pose_parameters = []
        for name, param in self.model.module.named_parameters(recurse=True):
            if name.split(".")[0] == "unet":
                continue
            pose_parameters.append(param)
        pose_parameters += list(pose_criterion.parameters())

        pose_optimizer = self.params.pose_optimizer(
            params=pose_parameters,
            lr=self.params.pose_learning_rate,
            weight_decay=self.params.pose_weight_decay,
        )
        if self.params.segpose_model_cpt:
            pose_optimizer.load_state_dict(cpt["pose_optimizer"])
        if not self.params.train_pose:
            for p in pose_parameters:
                p.requires_grad = False
        '''
        if self.params.segpose_model_cpt:
            cpt = torch.load(params.segpose_model_cpt)
            unet_optimizer.load_state_dict(cpt["unet_optimizer"])
        if not params.train_unet:
            for p in unet_params:
                p.requires_grad = False

        return unet_optimizer, None


if __name__ == "__main__":

    config_file = "./config/synth/config.json"
    params = Params()
    params.config_file = config_file
    params.__post_init__()

    # Set the device
    device = "cuda"

    # start training
    unet = UNet(1, 1, True)
    segpose_model = SegPoseNet(unet, device=device, droprate=0.5, feat_dim=2048)
    segpose_model = nn.DataParallel(segpose_model).to(device)

    trainer = Trainer(segpose_model, params)
    trainer.train()

'''
def eval_step(self):
    # Evaluation

    writer.add_scalar("epoch loss", epoch_loss, epoch)
    model.eval()
    val_loss = eval_seg_net(model, val_loader, is_segpose=True)
    model.train()
    print(f'Epoch: {epoch} Train IOU Loss: {epoch_loss.item() / len(train_loader)}')
    print(f'Epoch: {epoch}  DiceCoeff IOU Loss: {val_loss.item()}')
    val_losses.append(val_loss)

    writer.add_scalar("DiceCoeff IOU : ", val_loss, step)

    writer.add_images(
        "event frame",
        ev_frame.view(-1, *ev_frame.size()[2:]).unsqueeze(1),
        step,
    )
    writer.add_images("masks-gt", mask_gt, step)
    writer.add_images(
        "masks-pred-probs", mask_pred, step,
    )
    writer.add_images(
        "masks-pred",
        torch.sigmoid(mask_pred) > params.threshold_conf,
        step,
    )
'''

