# AI-assisted implementation
# Reviewed and validated by Mincong Fu

import torch
import torch.nn as nn


class CNNOC(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(4, 6)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=(1, 6)), nn.MaxPool2d((1, 4)), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(1, 6)), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=(1, 6)), nn.MaxPool2d((1, 4)), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(1, 6)), nn.MaxPool2d((1, 4)), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Flatten()
        )

        self.fc_flatten = nn.Sequential(
            nn.Linear(1824,512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128,32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.5),
        )

        # target task head: NOC classification (5 classes)
        self.mixnum_head = nn.Linear(32, 5)

    def forward(self, X):

        x = self.features(X)
        x = self.fc_flatten(x)

        # predict NOC
        mixnum_out = self.mixnum_head(x).squeeze(1)   # (batch, num_classes)

        return mixnum_out


if __name__ == '__main__':
    X = torch.randn(50, 1, 6, 5000)
    net = CNNOC()
    noc = net(X)
    print(noc.shape)

    import torch

    def register_hooks(model):

        def hook_fn(module, inp, out):
            class_name = module.__class__.__name__
            if isinstance(out, torch.Tensor):
                print(f"{class_name:20} | output shape: {list(out.shape)}")
            elif isinstance(out, (list, tuple)):
                print(f"{class_name:20} | output shape: {[o.shape for o in out if hasattr(o,'shape')]}")

        for m in model.modules():
            if not list(m.children()):
                m.register_forward_hook(hook_fn)

    model = CNNOC()
    register_hooks(model)

    x = torch.randn(1, 1, 6, 5000)
    out = model(x)