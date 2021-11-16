import numpy as np
import torch
from torch.utils.data import Dataset
import os
import cv2
import NetworkWrapper
import itertools


class ImageDataSet(Dataset):
    def __init__(self, mode="train", transform=None, pre_transform=None, data_dir='Data'):
        self.pre_transform = pre_transform
        self.transform = transform
        self.dir = data_dir
        self.images, self.labels = [], []

        path = f"{self.dir}/{mode}"
        classes = os.listdir(path)
        class_dictionary = NetworkWrapper.DefinedNN.classes_dictionary()

        for cls in classes:
            class_path = f"{path}/{cls}"
            for filename in os.listdir(class_path):
                img = cv2.imread(f"{class_path}/{filename}", cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.images.append([np.float32(img) / 1.0])
                    self.labels.append(class_dictionary[cls])


        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.array(self.labels)

        permutation = np.random.permutation(len(self.images))
        self.images = self.images[permutation]
        self.labels = self.labels[permutation]

        if self.pre_transform:
            print("Pre-Transforming Data")
            self.images = [self.pre_transform(image) for image in self.images]
            self.images = torch.stack(self.images)
            print("Finished Pre-Transforming.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.images[idx, :]
        labels = self.labels[idx]
        if self.transform:
            # sample = sample / 255.0
            sample = self.transform(sample)
            # sample = sample * 255.0
        return sample, labels


class OpenCVDataset(Dataset):
    def __init__(self, mode="train", transform=None, pre_transform=None, data_dir='processed', min_confidence=0, filter_success=False, col_ranges=None):
        self.pre_transform = pre_transform
        self.transform = transform
        self.dir = data_dir
        self.image_data, self.labels = [], []

        path = f"{self.dir}/{mode}"
        class_dictionary = NetworkWrapper.DefinedNN.classes_dictionary()

        for key in class_dictionary:
            class_path = f"{path}/{key}.csv"

            if col_ranges:
                with open(class_path) as f:
                    headers = f.readline()
                    headers = headers.split(',')
                    headers[-1] = headers[-1].strip("\n")

                range_array = []
                for col_range in col_ranges:
                    start = headers.index(col_range[0])
                    stop = headers.index(col_range[1])
                    range_array.append(range(start, stop+1, 1))

                desired_cols = itertools.chain(*range_array)
            else:
                desired_cols = None

            data = np.loadtxt(class_path, delimiter=',', dtype=None, skiprows=1, usecols=desired_cols)
            if filter_success:
                success = np.loadtxt(class_path, delimiter=',', dtype=None, skiprows=1, usecols=4)
                indexing = success == 1
            else:
                confidence = np.loadtxt(class_path, delimiter=',', dtype=None, skiprows=1, usecols=3)
                indexing = confidence >= min_confidence

            data = data[indexing]
            self.image_data.append([np.float32(data[0:]) / 1.0])
            self.labels.append(np.full(len(data), class_dictionary[key]))

        self.image_data = np.concatenate(self.image_data, axis=1).squeeze()
        self.labels = np.concatenate(self.labels, axis=0)

        permutation = np.random.permutation(len(self.image_data))
        self.image_data = self.image_data[permutation]
        self.labels = self.labels[permutation]
        print(self.image_data.shape)
        if self.pre_transform:
            print("Pre-Transforming Data")
            self.image_data = [self.pre_transform(image) for image in self.image_data]
            self.image_data = torch.stack(self.image_data)
            print("Finished Pre-Transforming.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.image_data[idx, :]
        labels = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, labels

if __name__ == '__main__':
    import torchvision.transforms as transforms

    train_set = OpenCVDataset(mode="train",
                              transform=None,
                              pre_transform=torch.Tensor,
                              data_dir="processed",
                              col_ranges=[[" gaze_0_x", " gaze_angle_y"], [" AU01_r"," AU45_c"]])
