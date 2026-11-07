#%%
# AI-assisted implementation
# Reviewed and validated by Mincong Fu

import torch
import torch.nn as nn


class CNNOC(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            # Input: (B, 1, 24, 80)
            nn.Conv2d(
                1, 32,
                kernel_size=(3, 5),
                padding=(1, 2)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                32, 32,
                kernel_size=(3, 3),
                padding=(1, 1)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Compress only the allele dimension while preserving the marker dimension
            nn.MaxPool2d(kernel_size=(1, 2)),

            # Output shape: (B, 32, 24, 40)
            nn.Conv2d(
                32, 64,
                kernel_size=(3, 3),
                padding=(1, 1)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 64,
                kernel_size=(1, 3),
                padding=(0, 1)
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Shape: (B, 64, 24, 40) -> (B, 64, 24, 20)
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(
                64, 32,
                kernel_size=(1, 3),
                padding=(0, 1)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Fix the output dimensions to support different input widths
            nn.AdaptiveAvgPool2d((24, 8)),
            # Output: (B, 32, 24, 8)
        )

        self.fc_flatten = nn.Sequential(
            nn.Flatten(),

            # 32 * 24 * 8 = 6144
            nn.Linear(32 * 24 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.mixnum_head = nn.Linear(32, num_classes)

    def forward(self, X):
        # Input: (B, 1, 24, 80)
        x = self.features(X)
        x = self.fc_flatten(x)

        # Return classification logits: (B, num_classes)
        return self.mixnum_head(x)


if __name__ == "__main__":
    model = CNNOC(num_classes=5)

    # Generate 50 synthetic samples
    X = torch.randn(50, 1, 24, 80)

    logits = model(X)

    print("Input shape:", X.shape)
    print("Output shape:", logits.shape)

    assert logits.shape == (50, 5)

    # Inspect the output shape after each convolutional stage
    with torch.no_grad():
        x = X
        for layer in model.features:
            x = layer(x)
            print(
                f"{layer.__class__.__name__:20s}: "
                f"{tuple(x.shape)}"
            )