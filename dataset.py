import os
import numpy as np
import torch
import glob
import tifffile

from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image

from utils import *



class N2NSliceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.file_list_by_folder = self.get_file_list_by_folder(root_folder_path)
        self.pairs, self.cumulative_slices = self.make_pairs()

    def get_file_list_by_folder(self, root_folder_path):
        file_list_by_folder = {}
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            file_paths = [os.path.join(subdir, f) for f in sorted_files]
            if file_paths:
                file_list_by_folder[subdir] = file_paths
        return file_list_by_folder

    def make_pairs(self):
        pairs = []
        cumulative_slices = [0]  # Initialize with 0 to start cumulative count
        for folder, files in self.file_list_by_folder.items():
            for f in files:
                volume = tifffile.imread(f)
                num_slices = volume.shape[0] - 1  # Subtract 1 because we form pairs within the same volume
                for i in range(num_slices):
                    # Each pair is (current_slice, next_slice) in the same volume
                    pairs.append((f, i, i + 1))  # Store filename and slice indices
                # Update cumulative slices count
                cumulative_slices.append(cumulative_slices[-1] + num_slices)
        return pairs, cumulative_slices

    def __len__(self):
        # The total number of pairs is the last element in the cumulative_slices list
        return self.cumulative_slices[-1]

    def __getitem__(self, index):
        # Find which pair the index falls into
        pair_index = next(i for i, total in enumerate(self.cumulative_slices) if total > index) - 1
        # Get the corresponding file and slice indices
        file_path, slice_index_1, slice_index_2 = self.pairs[pair_index]
        
        volume = tifffile.imread(file_path)
        input_slice = volume[slice_index_1, :, :]
        target_slice = volume[slice_index_2, :, :]

        input_slice_final = input_slice[..., None]
        target_slice_final = target_slice[..., None]

        if self.transform:
            data = self.transform((input_slice_final, target_slice_final))

        return data
    



class N2N4InputSliceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.file_list_by_folder = self.get_file_list_by_folder(root_folder_path)
        self.pairs, self.cumulative_slices = self.make_pairs()

    def get_file_list_by_folder(self, root_folder_path):
        file_list_by_folder = {}
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            file_paths = [os.path.join(subdir, f) for f in sorted_files]
            if file_paths:
                file_list_by_folder[subdir] = file_paths
        return file_list_by_folder

    def make_pairs(self):
        pairs = []
        cumulative_slices = [0]
        for folder, files in self.file_list_by_folder.items():
            for f in files:
                volume = tifffile.imread(f)
                num_slices = volume.shape[0]
                if num_slices >= 5:  # Ensure at least 5 slices for forming pairs
                    for i in range(num_slices - 4):
                        input_slices_indices = [i, i+1, i+3, i+4]
                        target_slice_index = i + 2
                        pairs.append((f, input_slices_indices, target_slice_index))
                        cumulative_slices.append(cumulative_slices[-1] + 1)
        return pairs, cumulative_slices

    def __len__(self):
        return self.cumulative_slices[-1]

    def __getitem__(self, index):
        pair_index = next(i for i, total in enumerate(self.cumulative_slices) if total > index) - 1
        file_path, input_slice_indices, target_slice_index = self.pairs[pair_index]
        
        volume = tifffile.imread(file_path)
        input_slices = np.stack([volume[i] for i in input_slice_indices], axis=-1)  # Stack slices along the last dimension
        target_slice = volume[target_slice_index][..., np.newaxis]  # Add a new axis to match input shape

        if self.transform:
            input_slices, target_slice = self.transform((input_slices, target_slice))

        return input_slices, target_slice



class PPSDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.file_list_by_folder = self.get_file_list_by_folder(root_folder_path)
        self.slices = self.make_slices()

    def get_file_list_by_folder(self, root_folder_path):
        file_list_by_folder = {}
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            file_paths = [os.path.join(subdir, f) for f in sorted_files]
            if file_paths:
                file_list_by_folder[subdir] = file_paths
        return file_list_by_folder

    def make_slices(self):
        slices = []
        for folder, files in self.file_list_by_folder.items():
            for f in files:
                volume = tifffile.imread(f)
                num_slices = volume.shape[0]
                if num_slices >= 4:  # Ensure at least 4 slices to form a stack
                    for i in range(num_slices - 3):  # Adjust for continuous slices
                        input_slices_indices = [i, i+1, i+2, i+3]  # Adjust to get continuous slices
                        slices.append((f, input_slices_indices))
        return slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        file_path, input_slice_indices = self.slices[index]
        
        volume = tifffile.imread(file_path)
        input_slices = np.stack([volume[i] for i in input_slice_indices], axis=-1)  # Stack slices along the last dimension

        if self.transform:
            input_slices = self.transform(input_slices)

        return input_slices




class N2NInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, transform=None, mean=None, std=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.mean = mean
        self.std = std
        self.file_list = self.get_file_list(root_folder_path)
        self.cumulative_slices = self.get_cumulative_slices()

    def get_file_list(self, root_folder_path):
        file_list = []
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            file_paths = [os.path.join(subdir, f) for f in sorted_files]
            file_list.extend(file_paths)
        return file_list

    def get_cumulative_slices(self):
        cumulative_slices = [0]
        for img_path in self.file_list:
            volume = tifffile.imread(img_path)
            cumulative_slices.append(cumulative_slices[-1] + volume.shape[0])
        return cumulative_slices

    def __len__(self):
        # The total number of slices is the last element in the cumulative_slices list
        return self.cumulative_slices[-1]

    def __getitem__(self, index):
        # Find which image stack the index falls into
        stack_index = next(i for i, total in enumerate(self.cumulative_slices) if total > index) - 1
        # Adjust index based on the start of the current stack
        adjusted_index = index - self.cumulative_slices[stack_index]
        
        img_path = self.file_list[stack_index]
        volume = tifffile.imread(img_path)

        # Extract the specific slice from the volume
        slice_ = volume[adjusted_index, :, :]

        # Add channel dimension if necessary
        slice_final = slice_[..., None]

        if self.transform:
            slice_final = self.transform(slice_final)

        return slice_final

