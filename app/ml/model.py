"""
TideTrace Deep Learning SAR Segmentation Model (U-Net Architecture).
3-Class Semantic Segmentation:
  - Class 0: Sea / Ambient Water
  - Class 1: Look-alike (natural biogenic films, calm water zones, internal waves)
  - Class 2: Mineral Oil (crude petroleum discharge, bunker fuel, bilge wash)
Model parameters strictly maintained under 80 MB limit (~30.8 MB FP32 footprint).
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2 with residual addition."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x) + self.residual(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_x != 0 or diff_y != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping feature maps to class logits."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TideTraceUNet(nn.Module):
    """
    TideTrace 3-Class Deep SAR Segmentation U-Net.
    Compact, high-resolution feature extractor with multi-scale skip connections.
    Total parameters: ~7.7 Million (<32 MB FP32), well within the 80 MB limit.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_features: int = 32,
        bilinear: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, base_features)                           # 32
        self.down1 = Down(base_features, base_features * 2)                         # 64
        self.down2 = Down(base_features * 2, base_features * 4)                     # 128
        self.down3 = Down(base_features * 4, base_features * 8)                     # 256
        self.down4 = Down(base_features * 8, (base_features * 16) // factor)        # 512 // factor

        # Decoder (Expanding Path)
        self.up1 = Up(base_features * 16, (base_features * 8) // factor, bilinear)
        self.up2 = Up(base_features * 8, (base_features * 4) // factor, bilinear)
        self.up3 = Up(base_features * 4, (base_features * 2) // factor, bilinear)
        self.up4 = Up(base_features * 2, base_features, bilinear)
        
        # Classification Head
        self.outc = OutConv(base_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw class logits.
        Args:
            x: Input tensor of shape (B, C, H, W) normalized to [0, 1].
        Returns:
            Logits of shape (B, num_classes, H, W).
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized class probabilities via Softmax along channel dimension.
        Args:
            x: Input tensor (B, C, H, W).
        Returns:
            Probabilities (B, num_classes, H, W) summing to 1.0 per pixel.
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_model_size_mb(self) -> float:
        """Calculate total parameter memory size in megabytes (FP32)."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str | Path, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        """Save model state dict and architecture metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "bilinear": self.bilinear,
            "metadata": extra_meta or {},
        }
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: Optional[str | torch.device] = None,
    ) -> Union["TideTraceUNet", "StandardTideTraceUNet"]:
        """
        Load a trained checkpoint from disk.
        Supports both raw state_dict, nested checkpoint dictionaries, and StandardTideTraceUNet checkpoints.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        map_location = torch.device(device) if device else torch.device("cpu")
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            sd = checkpoint["model_state_dict"]
            arch = checkpoint.get("architecture", "")
            if arch == "tidetrace_unet_32" or "conv_up1.net.0.weight" in sd:
                model = StandardTideTraceUNet(in_channels=2, num_classes=3, base_features=32)
                model.load_state_dict(sd)
            else:
                in_channels = checkpoint.get("in_channels", 3)
                num_classes = checkpoint.get("num_classes", 3)
                bilinear = checkpoint.get("bilinear", True)
                model = cls(in_channels=in_channels, num_classes=num_classes, bilinear=bilinear)
                model.load_state_dict(sd)
        elif isinstance(checkpoint, dict):
            if "conv_up1.net.0.weight" in checkpoint:
                model = StandardTideTraceUNet(in_channels=2, num_classes=3, base_features=32)
                model.load_state_dict(checkpoint)
            else:
                model = cls(in_channels=3, num_classes=3)
                model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unrecognized checkpoint format at {path}")

        if device:
            model.to(device)
        model.eval()
        return model


class StandardTideTraceUNet(nn.Module):
    """
    Standard U-Net architecture matching remote training pipelines.
    Dual-polarization (VV/VH) 2-channel input to 3-class segmentation output.
    """

    def __init__(self, in_channels: int = 2, num_classes: int = 3, base_features: int = 32):
        super().__init__()
        bf = base_features
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_features = bf

        class SubDoubleConv(nn.Module):
            def __init__(self, in_ch: int, out_ch: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.net(x)

        self.inc = SubDoubleConv(in_channels, bf)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), SubDoubleConv(bf, bf * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), SubDoubleConv(bf * 2, bf * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), SubDoubleConv(bf * 4, bf * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), SubDoubleConv(bf * 8, bf * 8))

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up1 = SubDoubleConv(bf * 16, bf * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up2 = SubDoubleConv(bf * 8, bf * 2)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up3 = SubDoubleConv(bf * 4, bf)
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up4 = SubDoubleConv(bf * 2, bf)
        self.outc = nn.Conv2d(bf, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 3 and self.in_channels == 2:
            x = x[:, :2, :, :]
        elif x.shape[1] == 1 and self.in_channels == 2:
            x = torch.cat([x, x], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        u1 = self.conv_up1(torch.cat([x4, self.up1(x5)], dim=1))
        u2 = self.conv_up2(torch.cat([x3, self.up2(u1)], dim=1))
        u3 = self.conv_up3(torch.cat([x2, self.up3(u2)], dim=1))
        u4 = self.conv_up4(torch.cat([x1, self.up4(u3)], dim=1))
        return self.outc(u4)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def get_model_size_mb(self) -> float:
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

