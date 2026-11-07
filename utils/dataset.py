# AI-assisted implementation
# Reviewed and validated by Mincong Fu

import torch
import os
import glob
import random

import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import Sampler


# Dataset for simulated data
class CNNOCSimDataset(Dataset):
    def __init__(self, base_dir="/home/fumincong/sim_epg/epg_simu_1", transform=None, preload=False):
        """
        Args:
            base_dir: Path to the simulated data folder
            transform: Optional data transformations
            preload: Whether to preload the data into memory
        """
        self.base_dir = base_dir
        self.transform = transform
        self.preload = preload

        # get all .npy files
        self.file_list = sorted(glob.glob(os.path.join(base_dir, "*.npy")))

        if len(self.file_list) == 0:
            raise RuntimeError(f"No .npy files found in {base_dir}")

        if self.preload:
            print("Preloading all .npy files into memory...")
            self.data_list = []
            self.mix_nums = []

            for file_path in self.file_list:
                X, mix_num = self._load_file(file_path)
                self.data_list.append(X)
                self.mix_nums.append(mix_num)

    def _parse_mix_num(self, file_path):
        file_name = os.path.basename(file_path)
        mix_num = int(file_name.split("_")[0])
        return mix_num

    def _load_file(self, file_path):
        mix_num = self._parse_mix_num(file_path)

        # load npy
        X = np.load(file_path)
        X = torch.from_numpy(X).float()

        # keep (C, L), add channel dimension -> (1, C, L)
        if X.dim() != 2:
            raise ValueError(f"Expected (C, L), got {X.shape} in {file_path}")

        X = X.unsqueeze(0)

        # 1~5 to 0~4
        mix_num = torch.tensor(mix_num - 1, dtype=torch.long)

        return X, mix_num

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.preload:
            X = self.data_list[idx]
            mix_num = self.mix_nums[idx]
        else:
            file_path = self.file_list[idx]
            X, mix_num = self._load_file(file_path)

        if self.transform:
            X = self.transform(X)

        return X, mix_num


# Dataset for real data
class CNNOCRealDataset(Dataset):
    def __init__(self,
                 csv_file,
                 base_dir,
                 split="finetune",
                 max_len=5000,
                 transform=None,
                 preload=False,
                 crop_csv="GF_crop_info.csv"):

        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.base_dir = base_dir
        self.max_len = max_len
        self.transform = transform
        self.preload = preload
        self.split = split

        self.injection_map = {
            "5sec": 0,
            "10sec": 1,
            "15sec": 2,
            "20sec": 3,
            "25sec": 4
        }

        if os.path.exists(crop_csv):
            crop_df = pd.read_csv(crop_csv)
            self.crop_dict = dict(zip(crop_df["file"], crop_df["crop_start"]))
            print(f"Loaded {len(self.crop_dict)} crop entries")
        else:
            self.crop_dict = {}
            print("WARNING: crop csv not found, no cropping applied")

        if self.preload:
            print("Preloading all .pth files into memory...")
            self.data_list = []
            self.labels = []

            for _, row in self.df.iterrows():
                mix_num = int(row["mix_num"])

                donor_vec = np.zeros(50)  # 50 = total number of possible donors
                if mix_num == 1:
                    donor_vec[int(row["sample_dilution_treatment"][:2]) - 1] = 1
                else:
                    donors = [int(x) for x in row["sample_indices"].split("_")]
                    for donor in donors:
                        donor_vec[donor - 1] = 1

                fname = row["file"][:-3] + "pth"
                file_path = os.path.join(self.base_dir, str(mix_num), fname)

                X = torch.load(file_path).float()  # (C, L)

                crop_start = self.crop_dict.get(fname, 0)
                C, L = X.shape
                if crop_start >= L:
                    crop_start = 0

                X = X[:, crop_start:crop_start + self.max_len]

                C, L_new = X.shape
                if L_new < self.max_len:
                    X = F.pad(X, (0, self.max_len - L_new))

                X = X.unsqueeze(0)  # (1, C, max_len)

                self.data_list.append(X)
                self.labels.append({
                    "donor_vec": torch.tensor(donor_vec, dtype=torch.float32),
                    "mix_num": torch.tensor(mix_num - 1, dtype=torch.long)
                })

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.preload:
            X = self.data_list[idx]
            label = self.labels[idx]
            donor_vec = label["donor_vec"]
            mix_num = label["mix_num"]
        else:
            row = self.df.iloc[idx]
            mix_num_raw = int(row["mix_num"])

            donor_vec = np.zeros(50)
            if mix_num_raw == 1:
                donor_vec[int(row["sample_dilution_treatment"][:2]) - 1] = 1
            else:
                donors = [int(x) for x in row["sample_indices"].split("_")]
                for donor in donors:
                    donor_vec[donor - 1] = 1

            fname = row["file"][:-3] + "pth"
            file_path = os.path.join(self.base_dir, str(mix_num_raw), fname)

            X = torch.load(file_path).float()  # (C, L)

            crop_start = self.crop_dict.get(fname, 0)
            C, L = X.shape
            if crop_start >= L:
                crop_start = 0

            X = X[:, crop_start:crop_start + self.max_len]

            C, L_new = X.shape
            if L_new < self.max_len:
                X = F.pad(X, (0, self.max_len - L_new))

            X = X.unsqueeze(0)  # (1, C, max_len)
            donor_vec = torch.tensor(donor_vec, dtype=torch.float32)
            mix_num = torch.tensor(mix_num_raw - 1, dtype=torch.long)

        if self.transform:
            X = self.transform(X)

        return X, donor_vec, mix_num

class BalancedSampler(Sampler):
    def __init__(self, labels, num_per_class=300):
        self.labels = np.array(labels)
        self.num_per_class = num_per_class
        self.classes = np.unique(self.labels)

        # precompute indices for each class
        self.class_indices = {
            cls: np.where(self.labels == cls)[0].tolist()
            for cls in self.classes
        }

    def __iter__(self):
        sampled_indices = []

        for cls in self.classes:
            idxs = self.class_indices[cls]
            n = len(idxs)

            if n >= self.num_per_class:
                # case 1: enough samples → random sampling without replacement
                selected = random.sample(idxs, self.num_per_class)

            else:
                # case 2: insufficient samples → repeat with reshuffling
                selected = idxs.copy()

                remain = self.num_per_class - n

                # reshuffle and sample iteratively without replacement
                pool = idxs.copy()
                random.shuffle(pool)

                while remain > 0:
                    take = min(remain, len(pool))
                    selected.extend(pool[:take])
                    remain -= take

                    # reshuffle for another round
                    random.shuffle(pool)

            sampled_indices.extend(selected)

        # shuffle all sampled indices across classes
        random.shuffle(sampled_indices)

        return iter(sampled_indices)

    def __len__(self):
        return self.num_per_class * len(self.classes)


if __name__=="__main__":
    dataset1 = CNNOCSimDataset()
    X, mix_num = dataset1[0]
    print(len(dataset1))
    print(X.shape)
    print(mix_num)