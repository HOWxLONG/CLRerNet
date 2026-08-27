from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class LegacyLaneClassifier(nn.Module):
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


# Keep the public import used by existing scripts and old checkpoints.
LaneClassifier = LegacyLaneClassifier


class DepthwiseTemporalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        padding = 2 * int(dilation)
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=5,
                padding=padding,
                dilation=int(dilation),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class AppearanceProfile(nn.Module):
    """Extract smooth color, occupancy, continuity, gap, and transition profiles."""

    def __init__(self, bins=16):
        super().__init__()
        self.bins = int(bins)

    def forward(self, x):
        # Input crops are BGR tensors in [0, 1]. Smooth thresholds keep the
        # feature useful under exposure changes without making a hard decision.
        b, g, r = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        value = torch.maximum(torch.maximum(b, g), r)
        low = torch.minimum(torch.minimum(b, g), r)
        saturation = (value - low) / value.clamp_min(1e-4)
        white = torch.sigmoid((value - 0.58) * 12.0) * torch.sigmoid((0.38 - saturation) * 12.0)
        yellow_strength = torch.minimum(r, g) - b
        yellow_balance = 1.0 - (r - g).abs()
        yellow = (
            torch.sigmoid((yellow_strength - 0.12) * 12.0)
            * torch.sigmoid((value - 0.42) * 10.0)
            * yellow_balance.clamp(0.0, 1.0)
        )
        gray = 0.114 * b + 0.587 * g + 0.299 * r
        local_mean = F.avg_pool2d(gray, kernel_size=9, stride=1, padding=4)
        contrast = (gray - local_mean).abs().clamp(0.0, 1.0)
        contrast_response = torch.sigmoid((contrast - 0.05) * 18.0)
        marking = torch.maximum(torch.maximum(white, yellow), contrast_response * value)

        profiles = []
        for response in (white, yellow, contrast_response, marking):
            profile = F.adaptive_avg_pool2d(response, (self.bins, 1)).flatten(1)
            profiles.append(profile)
        marking_profile = profiles[-1]
        previous = torch.cat([marking_profile[:, :1], marking_profile[:, :-1]], dim=1)
        following = torch.cat([marking_profile[:, 1:], marking_profile[:, -1:]], dim=1)
        continuity = marking_profile * previous
        gap = previous * (1.0 - marking_profile) * following
        transition = (marking_profile - previous).abs()
        profiles.extend([continuity, gap, transition])
        return torch.cat(profiles, dim=1)


class SequenceFusionLaneClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.25, profile_bins=16):
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
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(192, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            DepthwiseTemporalBlock(128, dilation=1),
            DepthwiseTemporalBlock(128, dilation=2),
            DepthwiseTemporalBlock(128, dilation=4),
        )
        self.attention = nn.Conv1d(128, 1, kernel_size=1)
        self.appearance = AppearanceProfile(bins=profile_bins)
        self.appearance_encoder = nn.Sequential(
            nn.Linear(7 * int(profile_bins), 64),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout) * 0.5),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 + 64, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        feature = self.features(x).mean(dim=3)
        sequence = self.temporal(feature)
        weights = torch.softmax(self.attention(sequence), dim=2)
        attended = (sequence * weights).sum(dim=2)
        pooled = torch.cat([attended, sequence.mean(dim=2), sequence.amax(dim=2)], dim=1)
        appearance = self.appearance_encoder(self.appearance(x))
        return self.classifier(torch.cat([pooled, appearance], dim=1))


def build_model(num_classes=3, dropout=0.25, model_type='sequence_fusion'):
    model_type = str(model_type)
    if model_type == 'legacy':
        return LegacyLaneClassifier(num_classes=num_classes, dropout=dropout)
    if model_type == 'sequence_fusion':
        return SequenceFusionLaneClassifier(num_classes=num_classes, dropout=dropout)
    raise ValueError(f'Unsupported lane classifier model_type: {model_type}')


def load_classifier_checkpoint(checkpoint, device='cpu'):
    checkpoint = Path(checkpoint)
    data = torch.load(str(checkpoint), map_location=device)
    classes = tuple(data.get('classes', CLASSES))
    model_type = str(data.get('model_type', 'legacy'))
    model = build_model(
        num_classes=len(classes),
        dropout=float(data.get('dropout', 0.25)),
        model_type=model_type,
    )
    state_dict = data.get('state_dict', data.get('model', data))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    meta = {
        'classes': classes,
        'crop_size': tuple(data.get('crop_size', (288, 128))),
        'strip_width': int(data.get('strip_width', 128)),
        'crop_mode': str(data.get('crop_mode', 'fixed')),
        'strip_reference_width': int(data.get('strip_reference_width', 2560)),
        'strip_min_width': int(data.get('strip_min_width', 64)),
        'strip_max_width': int(data.get('strip_max_width', 192)),
        'normalized_size': tuple(data.get('normalized_size', (1024, 544))),
        'top_crop_ratio': float(data.get('top_crop_ratio', 0.08)),
        'model_type': model_type,
        'temperature': max(float(data.get('temperature', 1.0)), 1e-4),
        'metrics': data.get('metrics', {}),
        'epoch': data.get('epoch'),
    }
    return model, meta
