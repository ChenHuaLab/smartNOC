#%%
# AI-assisted implementation
# Reviewed and validated by Mincong Fu

import numpy as np
import torch
from torch.utils.data import Dataset


class CNNOCAlleleDataset(Dataset):
    """
    Input:
        X: (N, 24, W) or (N, 1, 24, W)
        y: (N,), where mix_num is either 1-5 or already encoded as 0-4

    Output for a single sample:
        X: (1, 24, W)
        y: Scalar long tensor containing the class index 0-4
    """

    def __init__(self, x_file, y_file):
        self.X = np.load(x_file).astype(np.float32)
        self.y = np.asarray(np.load(y_file)).reshape(-1)

        if self.X.ndim == 3:
            # Convert (N, 24, W) to (N, 1, 24, W)
            self.X = self.X[:, None, :, :]
        elif self.X.ndim != 4:
            raise ValueError(
                f"X must have shape (N, 24, W) or (N, 1, 24, W), "
                f"but got {self.X.shape}"
            )

        if self.X.shape[1] != 1 or self.X.shape[2] != 24:
            raise ValueError(
                f"Expected input shape (N, 1, 24, W), "
                f"but got {self.X.shape}"
            )

        if len(self.X) != len(self.y):
            raise ValueError(
                f"Number of samples in X and y does not match: "
                f"{len(self.X)} vs {len(self.y)}"
            )

        if not np.isfinite(self.X).all():
            raise ValueError(
                "X contains NaN or Inf values. "
                "Please check the preprocessing results."
            )

        # Support both label formats:
        # Original mix_num labels: 1, 2, 3, 4, 5 -> 0, 1, 2, 3, 4
        # Already encoded labels: 0, 1, 2, 3, 4 -> unchanged
        self.y = self.y.astype(np.int64)

        if len(self.y) > 0:
            y_min = self.y.min()
            y_max = self.y.max()

            if y_min >= 1 and y_max <= 5:
                self.y = self.y - 1
            elif y_min >= 0 and y_max <= 4:
                pass
            else:
                raise ValueError(
                    f"mix_num labels must be in the range 0-4 or 1-5, "
                    f"but the current range is [{y_min}, {y_max}]"
                )

        print(f"Loaded X: {self.X.shape}")
        print(f"Loaded y: {self.y.shape}")
        print(
            f"Label distribution: "
            f"{dict(zip(*np.unique(self.y, return_counts=True)))}"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return X, y