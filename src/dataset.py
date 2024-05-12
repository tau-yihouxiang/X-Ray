import glob
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.sparse import csr_matrix
import torch.nn.functional as F
import torchvision


class DiffusionDataset(Dataset):
    def __init__(self, root_dir, size, num_frames, near, far, phase="train"):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        # Define the path to the folder containing video frames
        self.base_folder = root_dir
        self.size = size
        self.near = near
        self.far = far
        self.num_frames = num_frames
        self.depth_paths = glob.glob(os.path.join(root_dir, "depths/**/*.npz"), recursive=True)
        sorted(self.depth_paths)
        if phase == "train":
            del self.depth_paths[::10]
            random.shuffle(self.depth_paths)
        elif phase == "val":
            self.depth_paths = self.depth_paths[::20]
        else:
            self.depth_paths = self.depth_paths
        self.num_samples = len(self.depth_paths)        

    def __len__(self):
        return self.num_samples

    def load_depths(self, depths_path):
        loaded_data = np.load(depths_path)

        loaded_sparse_matrix = csr_matrix((loaded_data['data'], loaded_data['indices'], loaded_data['indptr']), shape=loaded_data['shape'])

        original_shape = (16, 1+3+3, 256, 256)
        restored_array = loaded_sparse_matrix.toarray().reshape(original_shape)
        return restored_array
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'xray_lr' tensor of shape (16, channels, 320, 512).
        """
        try:
            sample = {}
            xray_path = self.depth_paths[idx]
            depth_path = xray_path
            depths = self.load_depths(depth_path)

            xray = torch.from_numpy(depths.copy()).float()[:self.num_frames]  # [8, 7, H, W]
            hit = (xray[:, 0:1] > 0).clone().float() * 2 - 1
            xray[:, 0] = (xray[:, 0] - self.near) / (self.far - self.near) * 2 - 1 # (0-0.6) / (2.4 - 0.6) = -0.3333
            xray[:, 1:4] = F.normalize(xray[:, 1:4], dim=1)
            xray[:, 4:7] = xray[:, 4:7] * 2 - 1
            xray = torch.cat([xray, hit], dim=1)
            
            sample["xray"] = torch.nn.functional.interpolate(xray, size=(self.size, self.size), mode="nearest")
            xray_lr = torch.nn.functional.interpolate(xray, size=(self.size // 4, self.size // 4), mode="nearest")
            sample["xray_lr"] = torch.nn.functional.interpolate(xray_lr, size=(self.size, self.size), mode="nearest")

            # read condition image
            image_path = depth_path.replace("depths", "images").replace(".npz", ".png")
            image_values_pil = Image.open(image_path)
            
            # filter
            _, _, _, mask = image_values_pil.split()
            depth = (depths[0, 0] > 0).astype(np.float32)
            mask = (np.array(mask.resize(depth.shape)) / 255 > 0.5).astype(np.float32)
            iou = (mask * depth).sum() / np.maximum(mask, depth).sum()
            assert iou > 0.7, f"iou: {iou}"

            image_values_pil = image_values_pil.convert("RGB")
            image_values = image_values_pil.resize((self.size * 8, self.size * 8), Image.BILINEAR)
            image_values = torchvision.transforms.ToTensor()(image_values) * 2 - 1
            sample["image_values"] = image_values
            sample["image_path"] = image_path
            return sample
        
        except Exception as e:
            # print("Error: ", e)
            return self.__getitem__((idx + 1) % self.num_samples)


class UpsamplerDataset(Dataset):
    def __init__(self, root_dir, size, num_frames, near, far, type="diffusion", phase="train"):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        # Define the path to the folder containing video frames
        self.base_folder = root_dir
        self.size = size
        self.near = near
        self.far = far
        self.num_frames = num_frames
        self.depth_paths = glob.glob(os.path.join(root_dir, "depths/**/*.npz"), recursive=True)
        sorted(self.depth_paths)
        if phase == "train":
            del self.depth_paths[::10]
            random.shuffle(self.depth_paths)
        elif phase == "val":
            self.depth_paths = self.depth_paths[::10]
        else:
            self.depth_paths = self.depth_paths
        self.num_samples = len(self.depth_paths)        

    def __len__(self):
        return self.num_samples

    def load_depths(self, depths_path):
        loaded_data = np.load(depths_path)
        loaded_sparse_matrix = csr_matrix((loaded_data['data'], loaded_data['indices'], loaded_data['indptr']), shape=loaded_data['shape'])
        original_shape = (16, 1+3+3, 256, 256)
        restored_array = loaded_sparse_matrix.toarray().reshape(original_shape)
        return restored_array
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'xray_lr' tensor of shape (16, channels, 320, 512).
        """
        try:
            sample = {}
            xray_path = self.depth_paths[idx]
            depth_path = xray_path
            depths = self.load_depths(depth_path)

            xray = torch.from_numpy(depths.copy()).float()[:self.num_frames]  # [8, 7, H, W]
            hit = (xray[:, 0:1] > 0).clone().float() * 2 - 1
            xray[:, 0] = (xray[:, 0] - self.near) / (self.far - self.near) * 2 - 1 # (0-0.6) / (2.4 - 0.6) = -0.3333
            xray[:, 1:4] = F.normalize(xray[:, 1:4], dim=1)
            xray[:, 4:7] = xray[:, 4:7] * 2 - 1
            xray = torch.cat([xray, hit], dim=1)
            
            sample["xray"] = torch.nn.functional.interpolate(xray, size=(self.size, self.size), mode="nearest")
            sample["xray_lr"] = torch.nn.functional.interpolate(xray, size=(self.size // 4, self.size // 4), mode="nearest")

            # read condition image
            image_path = depth_path.replace("depths", "images").replace(".npz", ".png")
            image_values_pil = Image.open(image_path)
            
            # filter
            _, _, _, mask = image_values_pil.split()
            depth = (depths[0, 0] > 0).astype(np.float32)
            mask = (np.array(mask.resize(depth.shape)) / 255 > 0.5).astype(np.float32)
            iou = (mask * depth).sum() / np.maximum(mask, depth).sum()
            assert iou > 0.7, f"iou: {iou}"

            image_values_pil = image_values_pil.convert("RGB")
            image_values = image_values_pil.resize((self.size * 2, self.size * 2), Image.BILINEAR)
            image_values = torchvision.transforms.ToTensor()(image_values) * 2 - 1
            sample["image_values"] = image_values
            sample["image_path"] = image_path
            return sample
        
        except Exception as e:
            # print("Error: ", e)
            return self.__getitem__((idx + 1) % self.num_samples)