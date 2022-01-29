import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, ConcatDataset
from torchvision import transforms, utils
import numpy as np
import os
from .seg_hdf5 import EventSegHDF5
import pdb
import torch
from pathlib import Path

class DataModule(pl.LightningDataModule):
    def __init__(self, 
                    dataset_path="/home/ubuntu/datasets/11-6/animals", 
                    batch_size=8, 
                    shuffle=True, 
                    num_workers=0, 
                    drop_last=True,
                    min_skip=1,
                    max_skip=10,
                    width=640,
                    height=480,
                    train_seg=False,
                    sample_pixels=False,
                    pos_ratio=0.5,
                    pixel_width=0.002,
                    num_prev_events=200000,
                    num_cur_events=200000,
                    num_samples=3000,
                    overfit=False,
                    image_to_mask=False,
                    image_to_contour=False,
                    dense_contour=False,
                    delta_t=-1,
                    sdf=False,
                    use_new_ace=False,
                    **kwargs):
        super().__init__()
        self.path = dataset_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.min_skip = min_skip
        self.max_skip = max_skip
        self.is_seg = train_seg
        self.width = width
        self.height = height
        self.sample_pixels = sample_pixels
        self.pixel_width = pixel_width
        self.pos_ratio = pos_ratio
        self.sdf = sdf
        self.overfit = overfit
        self.num_cur_events = num_cur_events
        self.num_prev_events = num_prev_events
        self.num_samples = num_samples
        self.delta_t = delta_t
        self.use_new_ace = use_new_ace
        self.image_to_mask = image_to_mask
        self.image_to_contour = image_to_contour
        self.dense_contour = dense_contour
        self.setup()

    @staticmethod
    def add_data_args(parent_parser):
        # data
        parent_parser.add_argument('--dataset_path', type=str, default='/Datasets/cwang/event_pose')
        parent_parser.add_argument('--num_cur_events', type=int, default=200000)
        parent_parser.add_argument('--num_prev_events', type=int, default=200000)
        parent_parser.add_argument('--num_samples', type=int, default=3000)
        parent_parser.add_argument('--pixel_width', type=float, default=0.002)
        parent_parser.add_argument('--pos_ratio', type=float, default=0.5)
        parent_parser.add_argument('--sdf', action="store_true") 
        parent_parser.add_argument('--use_tp', action="store_true") 
        parent_parser.add_argument('--overfit', action="store_true") 
        parent_parser.add_argument('--delta_t', type=float, default=-1) 
        parent_parser.add_argument('--sample_pixels', action="store_true")
        parent_parser.add_argument('--use_new_ace', action="store_true")
        return parent_parser

    def generate_paths(self):
        all_h5_files_train = []
        all_h5_files_test = []

        with open(os.path.join(self.path, "train_split.txt"), 'r') as train_file:
            for f in train_file.readlines():
                f = f.strip("\n")
                path = Path(self.path)
                '''
                if "h5" in f and os.path.exists(os.path.join(str(path)+\
                                        "_frames_masks_labeled", f.strip(".h5"))):
                '''
                if "h5" in f:
                    all_h5_files_train.append(os.path.join(self.path, f))

        with open(os.path.join(self.path, "test_split.txt"), 'r') as test_file:
            for f in test_file.readlines():
                f = f.strip("\n")
                path = Path(self.path)
                '''
                if "h5" in f and os.path.exists(os.path.join(str(path)+\
                                        "_frames_masks_labeled", f.strip(".h5"))):
                '''
                if "h5" in f:
                    all_h5_files_test.append(os.path.join(self.path, f))

        return all_h5_files_train, all_h5_files_test

    def setup(self, stage=None):

        # dummy data
        all_sequences_train, all_sequences_val = self.generate_paths()

        # train
        all_datasets_train = []
        all_datasets_val = []
        data_class = EventSegHDF5

        composed = None

        all_datasets_train = []
        all_datasets_val = []

        for i in range(len(all_sequences_train)):
            dataset_one = data_class(all_sequences_train[i],
                                    width=self.width,
                                    height=self.height,
                                    min_skip=self.min_skip,
                                    max_skip=self.max_skip,
                                    sample_pixels=self.sample_pixels,
                                    pixel_width=self.pixel_width,
                                    pos_ratio=self.pos_ratio,
                                    sdf=self.sdf,
                                    overfit=self.overfit,
                                    num_cur_events=self.num_cur_events,
                                    num_prev_events=self.num_prev_events,
                                    num_samples=self.num_samples,
                                    image_to_mask=self.image_to_mask,
                                    image_to_contour=self.image_to_contour,
                                    dense_contour=self.dense_contour,
                                    delta_t=self.delta_t,
                                    max_length=1 if self.overfit else -1,
                                    use_new_ace=self.use_new_ace,
                                    transform=composed)
            all_datasets_train.append(dataset_one)

        for i in range(len(all_sequences_val)):
            dataset_one = data_class(all_sequences_val[i],
                                    width=self.width,
                                    height=self.height,
                                    min_skip=self.min_skip,
                                    max_skip=self.max_skip,
                                    sample_pixels=self.sample_pixels,
                                    pixel_width=self.pixel_width,
                                    pos_ratio=self.pos_ratio,
                                    sdf=self.sdf,
                                    overfit=self.overfit,
                                    num_cur_events=self.num_cur_events,
                                    num_prev_events=self.num_prev_events,
                                    num_samples = self.num_samples,
                                    image_to_mask=self.image_to_mask,
                                    image_to_contour=self.image_to_contour,
                                    dense_contour=self.dense_contour,
                                    delta_t=self.delta_t,
                                    max_length=1 if self.overfit else -1,
                                    use_new_ace=self.use_new_ace,
                                    transform=composed)
            all_datasets_val.append(dataset_one)

        self.train_dataset = ConcatDataset(all_datasets_train)
        self.val_dataset = ConcatDataset(all_datasets_val)

    def train_dataloader(self):
        # loader
        def worker_init_fn(worker_id):
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle,
                                  num_workers=self.num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=self.num_workers,
                                  drop_last=True)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.val_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers,
                                  drop_last=True)
        return test_loader
