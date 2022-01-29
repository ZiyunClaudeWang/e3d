import numpy as np
import h5py
import copy
import torch
import pickle
import pdb
import os
import cv2
from .event_utils import gen_discretized_event_volume, normalize_event_volume
from scipy.spatial.transform import Rotation as R
from pathlib import Path, PurePath
import torchvision.transforms.functional as TF
from pykdtree.kdtree import KDTree
from utils import get_contour, draw_contours
import torch.nn.functional as F
def angle_diff(angle_a, angle_b):
    diff = angle_b - angle_a
    return (diff + np.pi) % (np.pi * 2) - np.pi

class EventSegHDF5(torch.utils.data.Dataset):
    def __init__(self, path,  
                        pose_dict_path="/Datasets/cwang/cvpr2022/v2_objects/poses.pkl",
                        width=640, 
                        height=480, 
                        num_input_channels=20,
                        max_length=-1):
        self.path = path
        self.pose_dict_path = pose_dict_path
        self.width = width
        self.height = height
        self.num_input_channels = num_input_channels
        self.loaded = False
        self.max_length = max_length if max_length > 0 else np.inf

    def query_image(self, image_dir, mask_files):

        # load all image files
        ts = []
        image_names = []
        for f in os.listdir(image_dir):
            ts.append(int(f.replace(".png", "")))
            image_names.append(os.path.join(image_dir, f))
        ts = np.array(ts)
        image_names = np.array(image_names)

        mask_ts = np.array([int(f.replace(".png", "")) for f in mask_files])
        
        kd_tree = KDTree(np.array(ts))
        dist, idx = kd_tree.query(mask_ts, k=1)
        image_names = image_names[idx]

        return image_names

    def load(self):

        self.file = h5py.File(self.path, 'r')
        path = PurePath(self.path)

        filename = path.name.replace(".h5", "")
        self.t = self.file['t']

        mask_folder = os.path.join(str(path.parent.parent), "new_masks")
        image_folder = str(path.parent) + "_frames"

        self.mask_dir = PurePath(PurePath(mask_folder), filename)
        self.image_folder = PurePath(PurePath(image_folder), filename)

        self.mask_files = []
        self.mask_to_pose = []
        names = []

        for pose_idx, f in enumerate(sorted(os.listdir(self.mask_dir))):
            if not f.endswith(".png"):
                continue
            ts = int(f.replace(".png", ""))
            if ts < self.t[100]*1000 or ts > self.t[-100]*1000:
                continue
            mask_f = os.path.join(self.mask_dir, f)
            image_f = os.path.join(self.image_folder, f)
            self.mask_files.append(mask_f)
            names.append(f)
            self.mask_to_pose.append(pose_idx)

        self.mask_files = np.array(self.mask_files)
        self.image_files = self.query_image(self.image_folder, names)

        # events
        self.x = self.file['x']
        self.y = self.file['y']
        self.p = self.file['p']
        self.t = self.file['t']

        self.mask_ts = []
        self.binary_masks = []

        for ff in self.mask_files:
            ts = int(ff.split('/')[-1].replace('.png', ''))
            self.mask_ts.append(ts)

        order = np.argsort(self.mask_ts)
        self.mask_ts = np.array(self.mask_ts)
        self.mask_ts = self.mask_ts[order]
        self.mask_files = self.mask_files[order]
        self.image_files = self.image_files[order]
        self.mask_to_pose = np.array(self.mask_to_pose)
        self.mask_to_pose = self.mask_to_pose[order]

        # match ts based on ts, self.t is in micro second,
        # mask_ts is in nanosecond
        kd_tree = KDTree(np.array(self.t))
        dist, idx = kd_tree.query(self.mask_ts/1000, k=1)
        self.mask_to_event = idx

        self.num_events = self.t.shape[0]
        self.loaded = True

        #  find the angles associated with each mask
        print("Reading pose file")
        with open(self.pose_dict_path, 'rb') as pk:
            pose_dict = pickle.load(pk)
        all_angles = np.array(self.file['pose_to_angle'])[self.mask_to_pose]
        keys = np.array(list(pose_dict.keys()))
        all_poses = []
        for i in range(all_angles.shape[0]):
            diff = np.abs(angle_diff(keys, all_angles[i]))
            idx = np.argmin(diff)
            all_poses.append(pose_dict[keys[idx]])
        self.all_poses = np.array(all_poses)

    def close(self):

        self.events = None
        self.num_events = None
        self.file.close()
        self.length = None
        self.loaded = False
        self.binary_masks = None
        self.mask_ts = None

    def __len__(self):
        if not self.loaded:
            self.load()
        length = copy.deepcopy(len(self.mask_files))
        #self.close()
        return length - 3

    def __getitem__(self, idx):

        if not self.loaded:
            self.load()

        data = {}
        half_size = np.array([[640, 480]]) / 2

        # get prev_events (only used for volumes)
        num_events = 100000
        prev_end_index = int(self.mask_to_event[idx])
        prev_start_index = np.maximum(prev_end_index-num_events, 0)
        events = self.get_events_between(prev_start_index, prev_end_index)

        curr_batch_events = events
        img_size = [self.height, self.width]
        pos_events = curr_batch_events[curr_batch_events[:, -1] > 0]
        neg_events = curr_batch_events[curr_batch_events[:, -1] < 0]
        
        image_pos = np.zeros(img_size[0] * img_size[1], dtype="uint8")
        image_neg = np.zeros(img_size[0] * img_size[1], dtype="uint8")

        '''
        print((pos_events[:, 0] + pos_events[:, 1] * img_size[1]).astype("int32").max())
        print((neg_events[:, 0] + neg_events[:, 1] * img_size[1]).astype("int32").max())
        '''

        np.add.at(
            image_pos,
            (pos_events[:, 0] + pos_events[:, 1] * img_size[1]).astype("int32"),
            pos_events[:, -1] ** 2,
        )
        np.add.at(
            image_neg,
            (neg_events[:, 0] + neg_events[:, 1] * img_size[1]).astype("int32"),
            neg_events[:, -1] ** 2,
        )
        image_rgb = np.stack(
                    [
                        image_pos.reshape(img_size),
                        np.zeros(img_size, dtype="uint8"),
                        image_neg.reshape(img_size),
                    ],
                    -1,
                )
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
        data['event_frame'] = self.preprocess_images(image_gray)
        # get mask
        ff = self.mask_files[idx]
        try:
            mask = cv2.resize(cv2.imread(ff, cv2.IMREAD_GRAYSCALE), (640, 480))
        except:
            print("Error file: ",  ff)
        data['mask'] = self.preprocess_images(mask)

        # find the poses
        data['gt_pose'] = self.all_poses[idx]
        data['gt_R'] = self.all_poses[idx][:3, :3]
        data['gt_T'] = self.all_poses[idx][:3, 3]
        return data

    def preprocess_images(self, img: np.ndarray) -> torch.Tensor:
        """Normalize and convert to torch"""
        if img.dtype == np.uint16:
            img = img.astype(np.uint8)

        if img.max() > 1:
            img = img / 255

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        torch_img = torch.from_numpy(img)
        torch_img = torch_img.permute(2, 0, 1).float()
        return torch_img

    def preprocess_events(self, events):
        # No events in this time window
        if events.shape[0] == 0:
            print("no events")
            pdb.set_trace()

        # subtract out min to get delta time instead of absolute
        events[:,2] -= np.min(events[:,2])

        # normalize the timestamps
        events[:, 2] = (events[:, 2] - events[:, 2].min()) \
                        / (events[:, 2].max() - events[:, 2].min())

        # convolution expects 4xN
        events = events.astype(np.float32)
        return events

    def get_events_between(self, start_ind, end_ind):
        if not self.loaded:
            self.load()
        #events = self.events[start_ind:end_ind,:]
        
        events_x = np.array(self.x[start_ind:end_ind])
        events_y = np.array(self.y[start_ind:end_ind])
        events_t = np.array(self.t[start_ind:end_ind])

        # sutract to avoid overflow
        events_t = (events_t - events_t.min()).astype(float) / 1e6
        events_p = np.array(self.p[start_ind:end_ind]).astype(int)

        # change to -1, 1 if originally 0, 1
        events_p[events_p < 0.1] = -1
        events = np.stack((events_x, events_y, events_t, events_p), axis=1)

        if events.shape[0] == 0:
            print(start_ind, end_ind)
            pdb.set_trace()
        events = self.preprocess_events(events)

        return events

