from pathlib import Path

import torch
import torch.nn as nn

from .crop import CLASSES


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LaneClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.25):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 192),
            ConvBlock(192, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout) * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(num_classes=3, dropout=0.25):
    return LaneClassifier(num_classes=num_classes, dropout=dropout)


def load_classifier_checkpoint(checkpoint, device='cpu'):
    checkpoint = Path(checkpoint)
    data = torch.load(str(checkpoint), map_location=device)
    classes = tuple(data.get('classes', CLASSES))
    model = build_model(num_classes=len(classes), dropout=float(data.get('dropout', 0.25)))
    state_dict = data.get('state_dict', data.get('model', data))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    meta = {
        'classes': classes,
        'crop_size': tuple(data.get('crop_size', (288, 128))),
        'strip_width': int(data.get('strip_width', 128)),
        'metrics': data.get('metrics', {}),
        'epoch': data.get('epoch'),
    }
    return model, meta
